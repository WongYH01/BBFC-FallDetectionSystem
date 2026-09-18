package com.bbfc.notification.out.storage.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public final class SupabaseRequests {

    private SupabaseRequests() {
    }

    public record CreateBucket(
            String id,
            String name,
            boolean isPublic,
            @JsonProperty("file_size_limit") long fileSizeLimit,
            @JsonProperty("allowed_mime_types") List<String> allowedMimeTypes) {

        @JsonProperty("public")
        public boolean isPublic() {
            return isPublic;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record UploadResponse(@JsonProperty("Key") String key) {
    }
}
