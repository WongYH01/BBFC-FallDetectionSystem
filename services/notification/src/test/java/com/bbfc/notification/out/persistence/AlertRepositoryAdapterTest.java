package com.bbfc.notification.out.persistence;


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
                new Confidence(0.67)
        );

        alertRepositoryAdapter.save(alert);

        Optional<Alert> reloaded = alertRepositoryAdapter.findByEventId(new EventId("FE-20260910-0001"));

        assertThat(reloaded).isPresent();
        assertThat(reloaded.get().eventId()).isEqualTo(alert.eventId());
        assertThat(reloaded.get().room()).isEqualTo(alert.room());
        assertThat(reloaded.get().confidence()).isEqualTo(alert.confidence());
        assertThat(reloaded.get().state()).isEqualTo(alert.state());
    }

}
