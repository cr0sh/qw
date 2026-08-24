use std::{fs::{self,OpenOptions},io::Write as _,path::{Path,PathBuf},time::{SystemTime,UNIX_EPOCH}};
use crate::{codec::ContentBlob,EntryKey};
#[derive(Debug,Clone)]pub struct ScannedEntry{pub key:EntryKey,pub manifest:Vec<u8>}
#[derive(Debug)]pub struct StoredEntry{pub key:EntryKey,pub manifest:Vec<u8>,pub blobs:Vec<ContentBlob>}
pub trait PersistentSnapshotStore:Send{fn scan(&mut self,namespace:&str,now:u64)->Result<Vec<ScannedEntry>,String>;fn load(&mut self,key:&EntryKey)->Result<Option<StoredEntry>,String>;fn put(&mut self,e:StoredEntry,expires:u64)->Result<(),String>;fn refresh(&mut self,key:&EntryKey,expires:u64)->Result<(),String>;fn remove(&mut self,key:&EntryKey)->Result<(),String>;}
pub struct FilesystemSnapshotStore{root:PathBuf}
impl FilesystemSnapshotStore{
 pub fn new(root:impl Into<PathBuf>)->Result<Self,String>{let r=root.into();fs::create_dir_all(r.join("entries")).map_err(|e|e.to_string())?;fs::create_dir_all(r.join("blobs")).map_err(|e|e.to_string())?;Ok(Self{root:r})}
 fn path(&self,k:&EntryKey)->Result<PathBuf,String>{let mut p=k.0.split('/');let n=p.next().ok_or("invalid cache key")?;let d=p.next().ok_or("invalid cache key")?;if p.next().is_some()||!safe(n)||!safe(d){return Err("cache key contains invalid path components".into())}Ok(self.root.join("entries").join(n).join(format!("{d}.json")))}
 fn blob(&self,d:&str)->PathBuf{self.root.join("blobs").join(d)}
 fn cleanup_orphans(&self)->Result<(),String>{
  let mut refs=std::collections::HashSet::new();
  if let Ok(names)=fs::read_dir(self.root.join("entries")){
   for ns in names.flatten(){
    if let Ok(entries)=fs::read_dir(ns.path()){
     for entry in entries.flatten(){
      let path=entry.path();
      let Some(name)=path.file_name().and_then(|x|x.to_str()) else { continue };
      if name.starts_with(".tmp-") || name.contains(".json.tmp-") {
       let _=fs::remove_file(path);
       continue;
      }
      if let Ok(bytes)=fs::read(&path){
       if let Ok(m)=serde_json::from_slice::<crate::Manifest>(&bytes){refs.extend(m.blob_sha256);}
      }
     }
    }
   }
  }
  if let Ok(blobs)=fs::read_dir(self.root.join("blobs")){
   for b in blobs.flatten(){
    let path=b.path();
    let Some(name)=path.file_name().and_then(|x|x.to_str()) else { continue };
    if name.starts_with(".tmp-") || name.contains(".tmp-") {
     let _=fs::remove_file(path);
    } else if !refs.contains(name) {
     let _=fs::remove_file(path);
    }
   }
  }
  Ok(())
 }
}
impl PersistentSnapshotStore for FilesystemSnapshotStore{
 fn scan(&mut self,ns:&str,_:u64)->Result<Vec<ScannedEntry>,String>{if !safe(ns){return Err("cache namespace is not path-safe".into())}let mut out=Vec::new();let dir=self.root.join("entries").join(ns);if let Ok(entries)=fs::read_dir(dir){for e in entries.flatten(){let path=e.path();if path.extension().and_then(|x|x.to_str())!=Some("json"){continue}let Some(stem)=path.file_stem().and_then(|x|x.to_str()).map(str::to_owned)else{continue};if let Ok(manifest)=fs::read(path){out.push(ScannedEntry{key:EntryKey(format!("{ns}/{stem}")),manifest});}}}self.cleanup_orphans()?;Ok(out)}
 fn load(&mut self,k:&EntryKey)->Result<Option<StoredEntry>,String>{let p=self.path(k)?;let Ok(manifest)=fs::read(&p)else{return Ok(None)};let m:crate::Manifest=serde_json::from_slice(&manifest).map_err(|e|e.to_string())?;let mut blobs=Vec::new();for d in m.blob_sha256{let b=fs::read(self.blob(&d)).map_err(|e|format!("failed to read cache blob {d}: {e}"))?;blobs.push(ContentBlob{sha256:d,bytes:b.into()});}Ok(Some(StoredEntry{key:k.clone(),manifest,blobs}))}
 fn put(&mut self,e:StoredEntry,_:u64)->Result<(),String>{let p=self.path(&e.key)?;fs::create_dir_all(p.parent().unwrap()).map_err(|e|e.to_string())?;for b in &e.blobs{if b.sha256.len()!=64||!b.sha256.bytes().all(|x|x.is_ascii_hexdigit()){return Err("invalid blob digest".into())}let q=self.blob(&b.sha256);if !q.exists(){write_synced(&q,&b.bytes)?;}}let tmp=p.with_extension(format!("tmp-{}",SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()));write_synced(&tmp,&e.manifest)?;fs::rename(tmp,p).map_err(|e|e.to_string())?;Ok(())}
 fn refresh(&mut self,k:&EntryKey,expires:u64)->Result<(),String>{let p=self.path(k)?;let mut v:serde_json::Value=serde_json::from_slice(&fs::read(&p).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;v["expires_at_unix_ms"]=expires.into();let tmp=p.with_extension("tmp");write_synced(&tmp,&serde_json::to_vec(&v).map_err(|e|e.to_string())?)?;fs::rename(tmp,p).map_err(|e|e.to_string())}
 fn remove(&mut self,k:&EntryKey)->Result<(),String>{match fs::remove_file(self.path(k)?){Ok(())=>self.cleanup_orphans(),Err(e)if e.kind()==std::io::ErrorKind::NotFound=>Ok(()),Err(e)=>Err(e.to_string())}}
}
fn write_synced(p:&Path,b:&[u8])->Result<(),String>{let mut f=OpenOptions::new().create_new(true).write(true).open(p).map_err(|e|e.to_string())?;f.write_all(b).map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())}
fn safe(s:&str)->bool{!s.is_empty()&&s.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')}
