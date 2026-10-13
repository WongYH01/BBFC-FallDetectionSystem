package com.bbfc.notification.in.rest;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.service.AttachClipService;
import com.bbfc.notification.in.rest.dto.ClipResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class ClipController {

    private static final String MP4 = "video/mp4";

    private final AttachClipService attachClipService;

    public ClipController(AttachClipService attachClipService) {
        this.attachClipService = attachClipService;
    }

    @PutMapping(path = "/events/{eventId}/clip", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ClipResponse> attach(
            @PathVariable String eventId,
            @RequestPart("file") MultipartFile file) {

        // Filename and type are ours, not the uploader's: the bucket only accepts video/mp4 and
        // the key is derived from the event, so an uploaded name never reaches storage.
        ClipContent content = new ClipContent(
                eventId + ".mp4", MP4, file.getSize(), file::getInputStream);

        boolean sentToTelegram = attachClipService.attach(new EventId(eventId), content);

        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .body(new ClipResponse(eventId, "stored", sentToTelegram));
    }
}
