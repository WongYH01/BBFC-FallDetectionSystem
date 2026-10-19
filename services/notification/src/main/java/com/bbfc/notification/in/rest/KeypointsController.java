package com.bbfc.notification.in.rest;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.service.AttachKeypointsService;
import com.bbfc.notification.in.rest.dto.KeypointsResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;

@RestController
public class KeypointsController {

    private record Format(byte[] magic, String contentType) {
    }

    // .npz is a zip archive of .npy arrays; parquet files open with "PAR1".
    private static final Map<String, Format> FORMATS = Map.of(
            "npz", new Format(new byte[] {'P', 'K', 3, 4}, "application/octet-stream"),
            "parquet", new Format("PAR1".getBytes(StandardCharsets.US_ASCII), "application/vnd.apache.parquet"));

    private final AttachKeypointsService attachKeypointsService;

    public KeypointsController(AttachKeypointsService attachKeypointsService) {
        this.attachKeypointsService = attachKeypointsService;
    }

    @PutMapping(path = "/events/{eventId}/keypoints", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<KeypointsResponse> attach(
            @PathVariable String eventId,
            @RequestPart("file") MultipartFile file) throws IOException {

        String extension = extensionOf(file.getOriginalFilename());
        Format format = FORMATS.get(extension);
        if (format == null || !startsWith(file, format.magic())) {
            throw new ResponseStatusException(HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Keypoints must be a .npz or .parquet file");
        }

        // Only the extension is taken from the uploader; the stored name comes from the event.
        ClipContent content = new ClipContent(
                eventId + "." + extension, format.contentType(), file.getSize(), file::getInputStream);

        attachKeypointsService.attach(new EventId(eventId), content);

        return ResponseEntity.status(HttpStatus.ACCEPTED).body(new KeypointsResponse(eventId, "stored"));
    }

    private static String extensionOf(String filename) {
        int dot = filename == null ? -1 : filename.lastIndexOf('.');
        return dot < 0 ? "" : filename.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    private static boolean startsWith(MultipartFile file, byte[] magic) throws IOException {
        try (InputStream in = file.getInputStream()) {
            return Arrays.equals(in.readNBytes(magic.length), magic);
        }
    }
}
