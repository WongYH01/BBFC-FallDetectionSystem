package com.bbfc.notification.out.storage;

import com.bbfc.notification.config.ClipConfig;
import com.bbfc.notification.config.GcsStorageConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.core.port.KeypointStore;
import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.BucketInfo;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

@Component
@ConditionalOnProperty(prefix = "clips", name = "store", havingValue = "gcs")
public class GcsClipStore implements ClipStore, KeypointStore {

    private static final Logger log = LoggerFactory.getLogger(GcsClipStore.class);
    private static final String MP4 = "video/mp4";
    private static final int CONFLICT = 409;

    private final Storage storage;
    private final String bucket;
    private final long maxBytes;

    public GcsClipStore(Storage storage, GcsStorageConfig config, ClipConfig clipConfig) {
        this.storage = storage;
        this.bucket = config.bucket();
        this.maxBytes = clipConfig.maxBytes();
    }

    @Override
    public String store(String roomId, EventId eventId, ClipContent content) {
        String key = ClipObjectKey.of(roomId, eventId);
        if (content.sizeBytes() > maxBytes) {
            throw new ClipStorageException(
                    "Clip for " + eventId.eventId() + " is " + content.sizeBytes() + " bytes, over the " + maxBytes + " limit");
        }

        BlobInfo blob = BlobInfo.newBuilder(bucket, key).setContentType(MP4).build();
        try (InputStream in = content.open()) {
            storage.createFrom(blob, in);
            return key;
        } catch (IOException | StorageException e) {
            throw new ClipStorageException(
                    "Cloud Storage rejected the clip for " + eventId.eventId() + ": " + e.getMessage(), e);
        }
    }

    @Override
    public String storeKeypoints(String roomId, EventId eventId, Optional<Outcome> outcome, ClipContent content) {
        String key = ClipObjectKey.keypoints(roomId, eventId, outcome, content.filename());
        BlobInfo blob = BlobInfo.newBuilder(bucket, key).setContentType(content.contentType()).build();
        try (InputStream in = content.open()) {
            storage.createFrom(blob, in);
            return key;
        } catch (IOException | StorageException e) {
            throw new ClipStorageException(
                    "Cloud Storage rejected the keypoints for " + eventId.eventId() + ": " + e.getMessage(), e);
        }
    }

    @Override
    public String labelKeypoints(String key, Outcome outcome) {
        if (!ClipObjectKey.isUnlabelled(key)) {
            return key;
        }
        String target = ClipObjectKey.labelled(key, outcome);
        try {
            storage.moveBlob(Storage.MoveBlobRequest.newBuilder()
                    .setSource(BlobId.of(bucket, key))
                    .setTarget(BlobId.of(bucket, target))
                    .build());
            return target;
        } catch (StorageException e) {
            throw new ClipStorageException("Cloud Storage could not move " + key + ": " + e.getMessage(), e);
        }
    }

    @Override
    public void ensureBucket() {
        try {
            storage.create(BucketInfo.of(bucket));
            log.info("Created Cloud Storage bucket {}", bucket);
        } catch (StorageException e) {
            if (e.getCode() == CONFLICT) {
                log.info("Cloud Storage bucket {} already exists", bucket);
            } else {
                throw new ClipStorageException("Could not create bucket " + bucket + ": " + e.getMessage(), e);
            }
        }
    }
}
