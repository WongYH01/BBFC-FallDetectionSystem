package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Optional;

import com.bbfc.notification.core.domain.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.bbfc.notification.core.port.AlertRepository;

@ExtendWith(MockitoExtension.class)
class DedupAlertServiceTest {

    @Mock 
    private AlertRepository alertRepository;

    private final Clock clock = Clock.fixed(Instant.parse("2026-09-16T10:00:00Z"), ZoneOffset.UTC);

    private DedupAlertService dedupAlertService;
    private EventId eventId;
    private RoomRef room;
    private Confidence confidence;

    @BeforeEach
    void setUp() {
        dedupAlertService = new DedupAlertService(
                alertRepository, new DispatchPolicy(java.time.Duration.ofSeconds(20)), clock);
        eventId = new EventId("FE-1");
        room = new RoomRef("room-12", "Block A - Room 12");
        confidence = new Confidence(0.67);
    }

    @Test 
    void newEventIdCreatesAndSavesAlert(){

        given(alertRepository.findByEventId(eventId)).willReturn(Optional.empty());
        given(alertRepository.save(any())).willAnswer(invocation -> invocation.getArgument(0));

        DedupAlertService.Result result = dedupAlertService.handleAlert(eventId, room, confidence);

        assertThat(result.created()).isTrue();
        assertThat(result.alert().eventId()).isEqualTo(eventId);
        assertThat(result.alert().state()).isEqualTo(AlertState.PENDING);
        assertThat(result.alert().dispatchDeadline()).contains(Instant.parse("2026-09-16T10:00:20Z"));
    }

    @Test
    void existingEventIdReturnsExistingWithoutSaving() {
        Alert existing = Alert.receive(eventId, room, confidence, clock.instant(), clock.instant());
        given(alertRepository.findByEventId(eventId)).willReturn(Optional.of(existing));

        DedupAlertService.Result result = dedupAlertService.handleAlert(eventId, room, confidence);

        assertThat(result.created()).isFalse();
        assertThat(result.alert()).isSameAs(existing);
        verify(alertRepository, never()).save(any());
    }
}
