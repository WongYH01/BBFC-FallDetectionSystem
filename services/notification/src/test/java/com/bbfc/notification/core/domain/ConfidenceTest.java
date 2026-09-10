package com.bbfc.notification.core.domain;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import org.junit.jupiter.api.Test;

class ConfidenceTest {
    @Test
    void rejectsBelowZero() {
        assertThatThrownBy(() -> new Confidence(-0.1))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsAboveOne() {
        assertThatThrownBy(() -> new Confidence(1.1))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void bandsLowMediumHigh() {
        assertThat(new Confidence(0.2).band()).isEqualTo(Confidence.Band.LOW);
        assertThat(new Confidence(0.6).band()).isEqualTo(Confidence.Band.MEDIUM);
        assertThat(new Confidence(0.9).band()).isEqualTo(Confidence.Band.HIGH);
    }
}
