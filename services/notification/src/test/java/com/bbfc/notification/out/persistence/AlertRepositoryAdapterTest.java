package com.bbfc.notification.out.persistence;


import java.time.Instant;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.data.jpa.test.autoconfigure.DataJpaTest;
import org.springframework.boot.flyway.autoconfigure.FlywayAutoConfiguration;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.domain.SkeletonClip;

@DataJpaTest
@Testcontainers
@ImportAutoConfiguration(FlywayAutoConfiguration.class)
class AlertRepositoryAdapterTest {
    @Container 
    @ServiceConnection 
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15");

    @Autowired
    private AlertJPARepository alertJpaRepository;

    private AlertRepositoryAdapter alertRepositoryAdapter;

    @BeforeEach
    void setUp(){
        alertRepositoryAdapter = new AlertRepositoryAdapter(alertJpaRepository, new AlertMapper());
    }

    @Test 
    void saveThenFindByEventIdSucceeds() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0001"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );

        alertRepositoryAdapter.save(alert);

        Optional<Alert> reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0001"));

        assertThat(reloaded).isPresent();
        assertThat(reloaded.get().eventId()).isEqualTo(alert.eventId());
        assertThat(reloaded.get().room()).isEqualTo(alert.room());
        assertThat(reloaded.get().confidence()).isEqualTo(alert.confidence());
        assertThat(reloaded.get().state()).isEqualTo(alert.state());
        assertThat(reloaded.get().createdAt()).isEqualTo(alert.createdAt());
    }

    @Test
    void messageIdsSurviveARoundTrip() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0002"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );
        alert.recordDispatchMessage(501L);
        alert.recordEscalationMessage(502L);
        alert.recordEscalationMessage(503L);
        alert.recordFollowUpMessage(504L);

        alertRepositoryAdapter.save(alert);
        Alert reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0002")).orElseThrow();

        assertThat(reloaded.dispatchMessageId()).contains(501L);
        assertThat(reloaded.escalationMessageIds()).containsExactly(502L, 503L);
        assertThat(reloaded.followUpMessageId()).contains(504L);
    }

    @Test
    void findByEventIdForUpdateReturnsTheAlert() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0004"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );
        alertRepositoryAdapter.save(alert);

        Optional<Alert> locked = alertRepositoryAdapter.findByEventIdForUpdate(new EventId("FE-20260910-0004"));

        assertThat(locked).isPresent();
        assertThat(locked.get().eventId()).isEqualTo(alert.eventId());
    }

    @Test
    void findByEventIdForUpdateIsEmptyForUnknownEventId() {
        assertThat(alertRepositoryAdapter.findByEventIdForUpdate(new EventId("FE-does-not-exist"))).isEmpty();
    }

    @Test
    void clipSurvivesARoundTrip() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0005"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );
        alert.attachClip(SkeletonClip.stored("room-12/FE-20260910-0005.mp4", Instant.parse("2026-09-16T10:05:00Z")));
        alert.recordClipDelivery("file-abc", 601L);

        alertRepositoryAdapter.save(alert);
        Alert reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0005")).orElseThrow();

        SkeletonClip clip = reloaded.clip().orElseThrow();
        assertThat(clip.storageKey()).isEqualTo("room-12/FE-20260910-0005.mp4");
        assertThat(clip.storedAt()).isEqualTo(Instant.parse("2026-09-16T10:05:00Z"));
        assertThat(clip.telegramFileId()).isEqualTo("file-abc");
        assertThat(clip.telegramMessageId()).isEqualTo(601L);
    }

    @Test
    void clipStoredButNotDeliveredSurvivesARoundTrip() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0006"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );
        alert.attachClip(SkeletonClip.stored("room-12/FE-20260910-0006.mp4", Instant.parse("2026-09-16T10:05:00Z")));

        alertRepositoryAdapter.save(alert);
        Alert reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0006")).orElseThrow();

        assertThat(reloaded.clip().orElseThrow().isDelivered()).isFalse();
    }

    @Test
    void emptyMessageIdArraySurvivesARoundTrip() {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260910-0003"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67),
                Instant.parse("2026-09-16T10:00:00Z")
        );

        alertRepositoryAdapter.save(alert);
        Alert reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0003")).orElseThrow();

        assertThat(reloaded.dispatchMessageId()).isEmpty();
        assertThat(reloaded.escalationMessageIds()).isEmpty();
        assertThat(reloaded.followUpMessageId()).isEmpty();
    }

}
