package com.bbfc.notification.core.domain;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class CallbackCommandTest {

    @Test
    void acknowledgeRoundTrips() {
        CallbackCommand command = new CallbackCommand.Acknowledge(new EventId("FE-20260918-0003"));

        String encoded = command.encode();

        assertThat(encoded).isEqualTo("ack:FE-20260918-0003");
        assertThat(CallbackCommand.parse(encoded)).isEqualTo(command);
    }

    @ParameterizedTest
    @CsvSource({
            "GENUINE,     triage:FE-20260918-0003:genuine",
            "FALSE_ALARM, triage:FE-20260918-0003:false_alarm"
    })
    void triageRoundTrips(Outcome outcome, String expected) {
        CallbackCommand command = new CallbackCommand.Triage(new EventId("FE-20260918-0003"), outcome);

        String encoded = command.encode();

        assertThat(encoded).isEqualTo(expected);
        assertThat(CallbackCommand.parse(encoded)).isEqualTo(command);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "",
            "   ",
            "nonsense",
            "ack",                                   // missing event id
            "ack:FE-1:extra",                        // too many segments
            "triage:FE-1",                           // missing outcome
            "triage:FE-1:maybe",                     // unknown outcome
            "triage:FE-1:genuine:extra"              // too many segments
    })
    void rejectsMalformedCallbackData(String raw) {
        assertThatThrownBy(() -> CallbackCommand.parse(raw))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsNullCallbackData() {
        assertThatThrownBy(() -> CallbackCommand.parse(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsEncodingBeyondTelegramsSixtyFourByteLimit() {
        // "triage:" + ":false_alarm" is 19 bytes of overhead, leaving 45 for the event id,
        // while event_id is VARCHAR(64) — so this is reachable, not theoretical.
        String tooLong = "FE-".repeat(16);
        CallbackCommand command = new CallbackCommand.Triage(new EventId(tooLong), Outcome.FALSE_ALARM);

        assertThatThrownBy(command::encode)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("64");
    }

    @Test
    void acceptsAnEventIdExactlyAtTheLimit() {
        String maxLength = "F".repeat(45);
        CallbackCommand command = new CallbackCommand.Triage(new EventId(maxLength), Outcome.FALSE_ALARM);

        assertThat(command.encode()).hasSize(CallbackCommand.MAX_BYTES);
    }
}
