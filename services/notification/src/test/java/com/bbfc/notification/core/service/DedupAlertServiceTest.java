package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;

@ExtendWith(MockitoExtension.class)
public class DedupAlertServiceTest {

    @Mock 
    private AlertRepository alertRepository;

    @Test 
    void newEventIdCreatesAndSavesAlert(){
        DedupAlertService dedupAlertService = new DedupAlertService(alertRepository);
        EventId eventId = new EventId("FE-1");
        RoomRef room = new RoomRef("room-12", "Block A - Room 12");
        Confidence confidence = new Confidence(0.67);

        given(alertRepository.findByEventId(eventId)).willReturn(Optional.empty());
        given(alertRepository.save(any())).willAnswer(invocation -> invocation.getArgument(0));

        DedupAlertService.Result result = dedupAlertService.handleAlert(eventId, room, confidence);

        assertThat(result.created()).isTrue();
        assertThat(result.alert().eventId()).isEqualTo(eventId);
    }

    @Test
    void existingEventIdReturnsExistingWithoutSaving() {
        DedupAlertService service = new DedupAlertService(alertRepository);
        EventId eventId = new EventId("FE-1");
        RoomRef room = new RoomRef("room-12", "Block A - Room 12");
        Confidence confidence = new Confidence(0.67);
        Alert existing = Alert.dispatch(eventId, room, confidence);

        given(alertRepository.findByEventId(eventId)).willReturn(Optional.of(existing));

        DedupAlertService.Result result = service.handleAlert(eventId, room, confidence);

        assertThat(result.created()).isFalse();
        assertThat(result.alert()).isSameAs(existing);
        verify(alertRepository, never()).save(any());
    }
}
