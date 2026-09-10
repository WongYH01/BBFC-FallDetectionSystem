package com.bbfc.notification.domain;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import org.junit.jupiter.api.Test;

class EventIdTest {
    @Test
    void acceptsValidId() {
        assertThat(new EventId("FE-1").eventId()).isEqualTo("FE-1");
    }

    @Test
    void rejectsBlank() {
        assertThatThrownBy(() -> new EventId("   "))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsNull() {
        assertThatThrownBy(() -> new EventId(null))
            .isInstanceOf(IllegalArgumentException.class);
    }
}
