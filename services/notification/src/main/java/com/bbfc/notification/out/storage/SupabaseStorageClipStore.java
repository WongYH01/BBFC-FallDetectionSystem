package com.bbfc.notification.out.storage;

import com.bbfc.notification.config.SupabaseStorageConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.out.storage.dto.SupabaseRequests.CreateBucket;
import com.bbfc.notification.out.storage.dto.SupabaseRequests.UploadResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.io.InputStream;
import java.util.List;

@Component
@ConditionalOnProperty(prefix = "clips", name = "store", havingValue = "supabase", matchIfMissing = true)
public class SupabaseStorageClipStore implements ClipStore {

    private static final Logger log = LoggerFactory.getLogger(SupabaseStorageClipStore.class);
    private static final MediaType MP4 = MediaType.valueOf("video/mp4");
    private static final long BUCKET_FILE_SIZE_LIMIT = 8_388_608L;

    private final RestClient restClient;
    private final String bucket;

    public SupabaseStorageClipStore(SupabaseStorageConfig config, RestClient.Builder restClientBuilder) {
        this.bucket = config.bucket();
        this.restClient = restClientBuilder
                .baseUrl(config.url())
                .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + config.serviceKey())
                .build();
    }

    @Override
    public String store(String roomId, EventId eventId, ClipContent content) {
        String key = ClipObjectKey.of(roomId, eventId);

        try {
            UploadResponse response = restClient.put()
                    .uri("/object/{bucket}/{room}/{event}.mp4", bucket, roomId, eventId.eventId())
                    .contentType(MP4)
                    .header(HttpHeaders.CONTENT_LENGTH, Long.toString(content.sizeBytes()))
                    .body(out -> {
                        try (InputStream in = content.open()) {
                            in.transferTo(out);
                        }
                    })
                    .retrieve()
                    .body(UploadResponse.class);

            if (response == null || response.key() == null) {
                throw new ClipStorageException("Supabase Storage returned no key for " + eventId.eventId());
            }
            return key;
        } catch (RestClientResponseException e) {
            throw new ClipStorageException(
                    "Supabase Storage rejected the clip for " + eventId.eventId() + ": " + e.getMessage(), e);
        }
    }

    @Override
    public void ensureBucket() {
        try {
            restClient.post()
                    .uri("/bucket")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(new CreateBucket(bucket, bucket, false, BUCKET_FILE_SIZE_LIMIT, List.of(MP4.toString())))
                    .retrieve()
                    .toBodilessEntity();
            log.info("Created Supabase Storage bucket {}", bucket);
        } catch (RestClientResponseException e) {
            // storage-api answers a duplicate with HTTP 400 and the real code in the body, so
            // the status alone cannot tell "already there" from a genuine bad request.
            if (e.getResponseBodyAsString().contains("BucketAlreadyExists")) {
                log.info("Supabase Storage bucket {} already exists", bucket);
            } else {
                throw new ClipStorageException("Could not create bucket " + bucket + ": " + e.getMessage(), e);
            }
        }
    }
}
