package com.bbfc.notification.core.domain;

import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SkeletonClipTest {

    private static final Instant STORED_AT = Instant.parse("2026-09-16T10:05:00Z");

    @Test
    void storedClipIsNotYetDelivered() {
        SkeletonClip clip = SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT);

        assertThat(clip.storageKey()).isEqualTo("room-12/FE-1.mp4");
        assertThat(clip.storedAt()).isEqualTo(STORED_AT);
        assertThat(clip.telegramFileId()).isNull();
        assertThat(clip.telegramMessageId()).isNull();
        assertThat(clip.isDelivered()).isFalse();
    }

    @Test
    void deliveredReturnsANewClipKeepingTheStoredDetails() {
        SkeletonClip stored = SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT);

        SkeletonClip delivered = stored.delivered("file-abc", 601L);

        assertThat(delivered.storageKey()).isEqualTo("room-12/FE-1.mp4");
        assertThat(delivered.storedAt()).isEqualTo(STORED_AT);
        assertThat(delivered.telegramFileId()).isEqualTo("file-abc");
        assertThat(delivered.telegramMessageId()).isEqualTo(601L);
        assertThat(delivered.isDelivered()).isTrue();
        assertThat(stored.isDelivered()).isFalse();
    }

    @Test
    void rejectsABlankStorageKey() {
        assertThatThrownBy(() -> SkeletonClip.stored("  ", STORED_AT))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsANullStorageKey() {
        assertThatThrownBy(() -> SkeletonClip.stored(null, STORED_AT))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsANullStoredAt() {
        assertThatThrownBy(() -> SkeletonClip.stored("room-12/FE-1.mp4", null))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
