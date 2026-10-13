package com.bbfc.notification.out.storage;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

abstract class ClipStoreContract {

    static final byte[] CLIP_BYTES = "fake-mp4-bytes".getBytes(StandardCharsets.UTF_8);

    abstract ClipStore store();

    abstract byte[] storedBytes(String key);

    static ClipContent clip() {
        return new ClipContent("FE-1.mp4", "video/mp4", CLIP_BYTES.length,
                () -> new ByteArrayInputStream(CLIP_BYTES));
    }

    @Test
    void storesTheClipUnderARoomAndEventKey() {
        String key = store().store("room-12", new EventId("FE-1"), clip());

        assertThat(key).isEqualTo("room-12/FE-1.mp4");
        assertThat(storedBytes(key)).isEqualTo(CLIP_BYTES);
    }

    @Test
    void rejectsARoomIdThatWouldEscapeTheBucket() {
        assertThatThrownBy(() -> store().store("../../etc", new EventId("FE-1"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void rejectsAnEventIdThatWouldEscapeTheBucket() {
        assertThatThrownBy(() -> store().store("room-12", new EventId("../../etc"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }
}
