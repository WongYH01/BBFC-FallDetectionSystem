package com.bbfc.notification.domain;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import org.junit.jupiter.api.Test;

class RoomRefTest {
    @Test
    void acceptsValidValues() {
        RoomRef room = new RoomRef("room-1", "Block A, Room 1");
        assertThat(room.roomId()).isEqualTo("room-1");
        assertThat(room.displayName()).isEqualTo("Block A, Room 1");
    }

    @Test
    void rejectsBlankRoomId() {
        assertThatThrownBy(() -> new RoomRef("   ", "Block A"))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsBlankDisplayName() {
        assertThatThrownBy(() -> new RoomRef("room-1", "   "))
            .isInstanceOf(IllegalArgumentException.class);
    }
}
