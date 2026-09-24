package com.bbfc.notification.core.service;

import com.bbfc.notification.config.ClipConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.ClipAlreadyAttachedException;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.domain.SkeletonClip;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipDelivery;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.core.port.NotificationChannel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

@ExtendWith(MockitoExtension.class)
class AttachClipServiceTest {

    private static final EventId EVENT_ID = new EventId("FE-1");
    private static final Instant NOW = Instant.parse("2026-09-16T10:05:00Z");
    private static final byte[] BYTES = "fake-mp4-bytes".getBytes(StandardCharsets.UTF_8);

    @Mock
    private AlertRepository alertRepository;
    @Mock
    private ClipStore clipStore;
    @Mock
    private ClipRecorder clipRecorder;
    @Mock
    private NotificationChannel notificationChannel;

    private AttachClipService service;

    @BeforeEach
    void setUp() {
        service = new AttachClipService(
                alertRepository, clipStore, clipRecorder, notificationChannel,
                new ClipConfig(8_388_608L), Clock.fixed(NOW, ZoneOffset.UTC));
    }

    private static ClipContent content(long sizeBytes) {
        return new ClipContent("FE-1.mp4", "video/mp4", sizeBytes, () -> new ByteArrayInputStream(BYTES));
    }

    private static Alert dispatchedAlert() {
        Alert alert = Alert.dispatch(
                EVENT_ID,
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.92),
                Instant.parse("2026-09-16T10:00:00Z"));
        alert.recordDispatchMessage(501L);
        return alert;
    }

    @Test
    void storesTheClipThenSendsItAsAReply() {
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(dispatchedAlert()));
        given(clipStore.store(eq("room-12"), eq(EVENT_ID), any())).willReturn("skeleton-clips/room-12/FE-1.mp4");
        given(notificationChannel.sendClipReply(eq(501L), any()))
                .willReturn(Optional.of(new ClipDelivery("file-abc", 601L)));

        boolean sent = service.attach(EVENT_ID, content(1024));

        assertThat(sent).isTrue();
        verify(clipRecorder).markStored(EVENT_ID, SkeletonClip.stored("skeleton-clips/room-12/FE-1.mp4", NOW));
        verify(clipRecorder).markDelivered(EVENT_ID, new ClipDelivery("file-abc", 601L));
    }

    @Test
    void aRejectedTelegramSendStillLeavesTheClipStored() {
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(dispatchedAlert()));
        given(clipStore.store(any(), any(), any())).willReturn("skeleton-clips/room-12/FE-1.mp4");
        given(notificationChannel.sendClipReply(any(Long.class), any())).willReturn(Optional.empty());

        boolean sent = service.attach(EVENT_ID, content(1024));

        assertThat(sent).isFalse();
        verify(clipRecorder).markStored(any(), any());
        verify(clipRecorder, never()).markDelivered(any(), any());
    }

    @Test
    void unknownEventIsRejectedBeforeAnythingIsUploaded() {
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.empty());

        assertThatThrownBy(() -> service.attach(EVENT_ID, content(1024)))
                .isInstanceOf(AlertNotFoundException.class);

        verifyNoInteractions(clipStore, notificationChannel, clipRecorder);
    }

    @Test
    void aSecondClipIsRejectedBeforeAnythingIsUploaded() {
        Alert alert = dispatchedAlert();
        alert.attachClip(SkeletonClip.stored("skeleton-clips/room-12/FE-1.mp4", NOW));
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(alert));

        assertThatThrownBy(() -> service.attach(EVENT_ID, content(1024)))
                .isInstanceOf(ClipAlreadyAttachedException.class);

        verifyNoInteractions(clipStore, notificationChannel, clipRecorder);
    }

    @Test
    void anOversizeClipIsRejectedBeforeAnythingIsUploaded() {
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(dispatchedAlert()));

        assertThatThrownBy(() -> service.attach(EVENT_ID, content(8_388_609L)))
                .isInstanceOf(ClipTooLargeException.class);

        verifyNoInteractions(clipStore, notificationChannel, clipRecorder);
    }

    @Test
    void aStorageFailurePropagatesAndNeverReachesTelegram() {
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(dispatchedAlert()));
        given(clipStore.store(any(), any(), any())).willThrow(new ClipStorageException("storage down"));

        assertThatThrownBy(() -> service.attach(EVENT_ID, content(1024)))
                .isInstanceOf(ClipStorageException.class);

        verifyNoInteractions(notificationChannel);
        verify(clipRecorder, never()).markStored(any(), any());
    }

    @Test
    void withoutAnAlertMessageTheClipIsStoredButNotSent() {
        Alert alert = Alert.dispatch(
                EVENT_ID,
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.92),
                Instant.parse("2026-09-16T10:00:00Z"));
        given(alertRepository.findByEventId(EVENT_ID)).willReturn(Optional.of(alert));
        given(clipStore.store(any(), any(), any())).willReturn("skeleton-clips/room-12/FE-1.mp4");

        boolean sent = service.attach(EVENT_ID, content(1024));

        assertThat(sent).isFalse();
        verify(clipRecorder).markStored(any(), any());
        verifyNoInteractions(notificationChannel);
    }
}
