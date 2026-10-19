package com.bbfc.notification.core.service;

import java.time.Clock;

import org.springframework.stereotype.Service;

import com.bbfc.notification.config.ClipConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.KeypointsAlreadyAttachedException;
import com.bbfc.notification.core.domain.KeypointsFile;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.KeypointStore;

/** Stores a fall's keypoint file for training, already labelled if the alert has been triaged. */
@Service
public class AttachKeypointsService {

    private final AlertRepository alertRepository;
    private final KeypointStore keypointStore;
    private final KeypointsRecorder keypointsRecorder;
    private final ClipConfig clipConfig;
    private final Clock clock;

    public AttachKeypointsService(
            AlertRepository alertRepository, KeypointStore keypointStore, KeypointsRecorder keypointsRecorder,
            ClipConfig clipConfig, Clock clock) {
        this.alertRepository = alertRepository;
        this.keypointStore = keypointStore;
        this.keypointsRecorder = keypointsRecorder;
        this.clipConfig = clipConfig;
        this.clock = clock;
    }

    /** @return the key the file was stored under. */
    public String attach(EventId eventId, ClipContent content) {
        Alert alert = alertRepository.findByEventId(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));

        if (alert.keypoints().isPresent()) {
            throw new KeypointsAlreadyAttachedException(eventId);
        }
        if (content.sizeBytes() > clipConfig.maxBytes()) {
            throw new ClipTooLargeException(content.sizeBytes(), clipConfig.maxBytes());
        }

        String key = keypointStore.storeKeypoints(alert.room().roomId(), eventId, alert.outcome(), content);
        keypointsRecorder.markStored(eventId, new KeypointsFile(key, clock.instant()));
        // Triage may have landed between the read above and markStored.
        keypointsRecorder.labelIfReady(eventId);
        return key;
    }
}
