UPDATE alerts
SET clip_storage_key = substring(clip_storage_key FROM length('skeleton-clips/') + 1)
WHERE clip_storage_key LIKE 'skeleton-clips/%';
