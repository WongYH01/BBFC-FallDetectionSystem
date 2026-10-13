package com.bbfc.notification.in.rest;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.service.DedupAlertService;
import com.bbfc.notification.in.rest.dto.EventRequest;
import com.bbfc.notification.in.rest.dto.EventResponse;

import jakarta.validation.Valid;

@RestController 
public class IngestController {
    private final DedupAlertService dedupAlertService;

    public IngestController(DedupAlertService dedupAlertService) {
        this.dedupAlertService = dedupAlertService;
    }

    @PostMapping("/events")
    public ResponseEntity<EventResponse> ingest(@Valid @RequestBody EventRequest request){
        DedupAlertService.Result result = dedupAlertService.handleAlert(
            new EventId(request.eventId()), 
            new RoomRef(request.roomId(), request.roomName()), 
            new Confidence(request.confidence())
        );

        String status = result.created() ? "accepted" : "duplicate";
        HttpStatus httpStatus = result.created() ? HttpStatus.ACCEPTED : HttpStatus.OK;

        return ResponseEntity.status(httpStatus)
            .body(new EventResponse(result.alert().eventId().eventId(), status));
    }
}
