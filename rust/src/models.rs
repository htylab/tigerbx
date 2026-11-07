//! Release models that are not embedded (e.g. DGM12): downloaded on first use into the TigerBx model cache, the
//! directory and sha256 table of the Python package (`lib_tool.get_model`), so the two share the files.
//!
//! Cache: `TIGERBX_MODEL_DIR` if set, else the platform cache (platformdirs `user_cache_dir('tigerbx')`) + `models`:
//! Linux `$XDG_CACHE_HOME/tigerbx/models` (default `~/.cache`), macOS `~/Library/Caches/tigerbx/models`, Windows
//! `%LOCALAPPDATA%\tigerbx\Cache\models`. Offline: put the file there (its sha256 must match).

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;

/// The Python package's model table: one place for the release sha256 (`'<name>.onnx': '<sha256>'` lines).
static MODELS_PY: &str = include_str!("../../tigerbx/core/models.py");
/// TigerBx `lib_tool.model_servers`, tried in order.
const SERVERS: [&str; 2] = [
    "https://github.com/htylab/tigerbx/releases/download/modelhub/",
    "https://data.mrilab.org/onnxmodel/dev/",
];
const MODEL_DIR_ENV: &str = "TIGERBX_MODEL_DIR";

/// Release sha256 of model file `name` (`models.SHA256`).
pub fn release_sha256(name: &str) -> Option<String> {
    let key = format!("'{name}':");
    MODELS_PY.lines().find_map(|l| {
        let rest = l.trim().strip_prefix(&key)?;
        let h = rest.trim().trim_end_matches(',').trim_matches('\'');
        (h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())).then(|| h.to_string())
    })
}

/// The model cache directory (see the module doc).
pub fn model_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os(MODEL_DIR_ENV).filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        env("LOCALAPPDATA").context("LOCALAPPDATA not set")?.join("tigerbx").join("Cache")
    } else if cfg!(target_os = "macos") {
        env("HOME").context("HOME not set")?.join("Library").join("Caches").join("tigerbx")
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache"))).context("HOME not set")?.join("tigerbx")
    };
    Ok(base.join("models"))
}

fn sha256_file(p: &std::path::Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Path of release model `name` (e.g. `mprage_dgm12_v002_opt.onnx`) in the cache, downloaded (and checked against its
/// release sha256) when it is missing or does not match. A file that does not match is kept as `<name>.bad`.
pub fn get(name: &str) -> Result<PathBuf> {
    let expected = release_sha256(name).with_context(|| format!("{name} is not a release model"))?;
    let dir = model_dir()?;
    let path = dir.join(name);
    if path.exists() {
        if sha256_file(&path)? == expected {
            return Ok(path);
        }
        eprintln!("{name} does not match the release file (sha256): moved to {name}.bad, downloading it again");
        std::fs::rename(&path, dir.join(format!("{name}.bad")))?;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create the model cache {}", dir.display()))?;
    let mut errors = Vec::new();
    for server in SERVERS {
        let url = format!("{server}{name}");
        println!("Downloading {url}");
        let tmp = dir.join(format!("{name}.{}.tmp", std::process::id()));
        let res = download(&url, &tmp).and_then(|_| {
            let got = sha256_file(&tmp)?;
            if got != expected {
                bail!("sha256 mismatch (incomplete or wrong file)");
            }
            std::fs::rename(&tmp, &path)?;
            Ok(())
        });
        match res {
            Ok(()) => return Ok(path),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                errors.push(format!("{url}: {e:#}"));
            }
        }
    }
    bail!("could not download {name} (offline: put the release file into {}):\n  {}", dir.display(), errors.join("\n  "))
}

fn download(url: &str, dst: &std::path::Path) -> Result<()> {
    let mut resp = ureq::get(url).call()?;
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut f = std::fs::File::create(dst)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        f.write_all(&buf[..n])?;
    }
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_sha256_reads_the_python_table() {
        let h = release_sha256("mprage_dgm12_v002_opt.onnx").unwrap();
        assert_eq!(h.len(), 64);
        assert!(release_sha256("no_such_model.onnx").is_none());
    }
}
