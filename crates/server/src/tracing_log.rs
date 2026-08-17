use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::SetGlobalDefaultError;
use tracing::{Event, Level, Metadata, Subscriber, level_filters::LevelFilter};

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static SPAN_STACK: RefCell<Vec<Id>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn init() -> Result<(), SetGlobalDefaultError> {
    tracing::subscriber::set_global_default(JsonSubscriber {
        spans: Mutex::new(HashMap::new()),
    })
}

struct JsonSubscriber {
    spans: Mutex<HashMap<u64, SpanData>>,
}

struct SpanData {
    view: SpanView,
    lineage: Vec<SpanView>,
    references: usize,
    started: Instant,
}

#[derive(Clone)]
struct SpanView {
    name: &'static str,
    target: &'static str,
    fields: Map<String, Value>,
}

impl Subscriber for JsonSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= Level::INFO
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::INFO)
    }

    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let id = Id::from_u64(NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed));
        let contextual_parent = SPAN_STACK.with(|stack| stack.borrow().last().cloned());
        let parent = attributes.parent().cloned().or(contextual_parent);
        let mut fields = Map::new();
        attributes.record(&mut JsonVisitor(&mut fields));
        let view = SpanView {
            name: attributes.metadata().name(),
            target: attributes.metadata().target(),
            fields,
        };
        let lineage = {
            let spans = self.spans.lock().expect("tracing span registry lock");
            parent
                .as_ref()
                .and_then(|parent| spans.get(&parent.into_u64()))
                .map(|parent| {
                    let mut lineage = parent.lineage.clone();
                    lineage.push(parent.view.clone());
                    lineage
                })
                .unwrap_or_default()
        };
        let path = {
            let mut path = lineage.clone();
            path.push(view.clone());
            path
        };
        self.spans
            .lock()
            .expect("tracing span registry lock")
            .insert(
                id.into_u64(),
                SpanData {
                    view,
                    lineage,
                    references: 1,
                    started: Instant::now(),
                },
            );
        emit(
            Level::INFO,
            attributes.metadata().target(),
            "span.created",
            Map::new(),
            &path,
        );
        id
    }

    fn record(&self, span: &Id, values: &Record<'_>) {
        if let Some(span) = self
            .spans
            .lock()
            .expect("tracing span registry lock")
            .get_mut(&span.into_u64())
        {
            values.record(&mut JsonVisitor(&mut span.view.fields));
        }
    }

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = Map::new();
        event.record(&mut JsonVisitor(&mut fields));
        let path = self.current_path();
        emit(
            *event.metadata().level(),
            event.metadata().target(),
            "event",
            fields,
            &path,
        );
    }

    fn enter(&self, span: &Id) {
        SPAN_STACK.with(|stack| stack.borrow_mut().push(span.clone()));
    }

    fn exit(&self, span: &Id) {
        SPAN_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            debug_assert_eq!(popped.as_ref(), Some(span));
        });
    }

    fn clone_span(&self, id: &Id) -> Id {
        if let Some(span) = self
            .spans
            .lock()
            .expect("tracing span registry lock")
            .get_mut(&id.into_u64())
        {
            span.references += 1;
        }
        id.clone()
    }

    fn try_close(&self, id: Id) -> bool {
        let closed = {
            let mut spans = self.spans.lock().expect("tracing span registry lock");
            let Some(span) = spans.get_mut(&id.into_u64()) else {
                return false;
            };
            span.references -= 1;
            if span.references != 0 {
                return false;
            }
            spans.remove(&id.into_u64())
        };
        if let Some(span) = closed {
            let mut fields = Map::new();
            fields.insert(
                "duration_ms".to_string(),
                json!(span.started.elapsed().as_secs_f64() * 1_000.0),
            );
            let mut path = span.lineage;
            path.push(span.view.clone());
            emit(Level::INFO, span.view.target, "span.closed", fields, &path);
        }
        true
    }
}

impl JsonSubscriber {
    fn current_path(&self) -> Vec<SpanView> {
        let current = SPAN_STACK.with(|stack| stack.borrow().last().cloned());
        let Some(current) = current else {
            return Vec::new();
        };
        self.spans
            .lock()
            .expect("tracing span registry lock")
            .get(&current.into_u64())
            .map(|span| {
                let mut path = span.lineage.clone();
                path.push(span.view.clone());
                path
            })
            .unwrap_or_default()
    }
}

struct JsonVisitor<'a>(&'a mut Map<String, Value>);

impl Visit for JsonVisitor<'_> {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_string(), Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), value.into());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.into());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().to_string(), json!(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0
            .insert(field.name().to_string(), Value::String(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            Value::String(format!("{value:?}")),
        );
    }
}

fn emit(
    level: Level,
    target: &str,
    kind: &'static str,
    fields: Map<String, Value>,
    spans: &[SpanView],
) {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let spans = spans
        .iter()
        .map(|span| {
            json!({
                "name": span.name,
                "target": span.target,
                "fields": span.fields,
            })
        })
        .collect::<Vec<_>>();
    eprintln!(
        "{}",
        json!({
            "timestamp_ms": timestamp_ms,
            "level": level.as_str(),
            "target": target,
            "kind": kind,
            "fields": fields,
            "spans": spans,
        })
    );
}
