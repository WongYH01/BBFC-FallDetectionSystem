package com.bbfc.notification.out.storage;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.core.port.KeypointStore;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

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

    static final byte[] KEYPOINT_BYTES = "PK\u0003\u0004fake-npz".getBytes(StandardCharsets.ISO_8859_1);

    KeypointStore keypointStore() {
        return (KeypointStore) store();
    }

    static ClipContent keypoints(String filename) {
        return new ClipContent(filename, "application/octet-stream", KEYPOINT_BYTES.length,
                () -> new ByteArrayInputStream(KEYPOINT_BYTES));
    }

    @Test
    void storesUntriagedKeypointsAsIncoming() {
        String key = keypointStore().storeKeypoints(
                "room-12", new EventId("FE-1"), Optional.empty(), keypoints("FE-1.npz"));

        assertThat(key).isEqualTo("keypoints/incoming/room-12/FE-1.npz");
        assertThat(storedBytes(key)).isEqualTo(KEYPOINT_BYTES);
    }

    @Test
    void storesTriagedKeypointsStraightUnderTheirLabel() {
        String key = keypointStore().storeKeypoints(
                "room-12", new EventId("FE-1"), Optional.of(Outcome.FALSE_ALARM), keypoints("FE-1.parquet"));

        assertThat(key).isEqualTo("keypoints/labelled/false_alarm/room-12/FE-1.parquet");
        assertThat(storedBytes(key)).isEqualTo(KEYPOINT_BYTES);
    }

    @Test
    void leavesAnAlreadyLabelledKeyAlone() {
        String key = "keypoints/labelled/genuine/room-12/FE-1.npz";

        assertThat(keypointStore().labelKeypoints(key, Outcome.GENUINE)).isEqualTo(key);
    }

    @Test
    void rejectsAKeypointsExtensionThatWouldEscapeTheBucket() {
        assertThatThrownBy(() -> keypointStore().storeKeypoints(
                "room-12", new EventId("FE-1"), Optional.empty(), keypoints("FE-1./../x")))
                .isInstanceOf(ClipStorageException.class);
    }
}
