package com.bbfc.notification.out.storage;

import com.bbfc.notification.config.ClipConfig;
import com.bbfc.notification.config.GcsStorageConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.ClipStore;
import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.BucketInfo;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageException;
import com.google.cloud.storage.contrib.nio.testing.LocalStorageHelper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class GcsClipStoreTest extends ClipStoreContract {

    private static final String BUCKET = "test-clips";
    private static final long MAX_BYTES = 8_388_608L;

    private Storage storage;
    private GcsClipStore store;

    @BeforeEach
    void setUp() {
        storage = LocalStorageHelper.getOptions().getService();
        store = new GcsClipStore(storage, new GcsStorageConfig(BUCKET), new ClipConfig(MAX_BYTES));
    }

    @Override
    ClipStore store() {
        return store;
    }

    @Override
    byte[] storedBytes(String key) {
        return storage.readAllBytes(BlobId.of(BUCKET, key));
    }

    @Test
    void storesTheClipAsMp4() {
        String key = store.store("room-12", new EventId("FE-1"), clip());

        assertThat(storage.get(BlobId.of(BUCKET, key)).getContentType()).isEqualTo("video/mp4");
    }

    @Test
    void rejectsAClipOverTheSizeLimitWithoutUploading() {
        GcsClipStore tight = new GcsClipStore(storage, new GcsStorageConfig(BUCKET), new ClipConfig(4));

        assertThatThrownBy(() -> tight.store("room-12", new EventId("FE-2"), clip()))
                .isInstanceOf(ClipStorageException.class);
        assertThat(storage.get(BlobId.of(BUCKET, "room-12/FE-2.mp4"))).isNull();
    }

    @Test
    void wrapsAnUploadFailure() throws IOException {
        Storage failing = mock(Storage.class);
        given(failing.createFrom(any(BlobInfo.class), any(InputStream.class)))
                .willThrow(new StorageException(403, "denied"));
        GcsClipStore failingStore = new GcsClipStore(failing, new GcsStorageConfig(BUCKET), new ClipConfig(MAX_BYTES));

        assertThatThrownBy(() -> failingStore.store("room-12", new EventId("FE-1"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void wrapsAClipThatCannotBeRead() {
        ClipContent unreadable = new ClipContent("FE-1.mp4", "video/mp4", 1, () -> {
            throw new IOException("gone");
        });

        assertThatThrownBy(() -> store.store("room-12", new EventId("FE-1"), unreadable))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void createsTheBucket() {
        Storage mockStorage = mock(Storage.class);
        GcsClipStore mockStore = new GcsClipStore(mockStorage, new GcsStorageConfig(BUCKET), new ClipConfig(MAX_BYTES));

        mockStore.ensureBucket();

        verify(mockStorage).create(BucketInfo.of(BUCKET));
    }

    @Test
    void toleratesAnExistingBucket() {
        Storage mockStorage = mock(Storage.class);
        given(mockStorage.create(any(BucketInfo.class))).willThrow(new StorageException(409, "exists"));
        GcsClipStore mockStore = new GcsClipStore(mockStorage, new GcsStorageConfig(BUCKET), new ClipConfig(MAX_BYTES));

        assertThatCode(mockStore::ensureBucket).doesNotThrowAnyException();
    }

    @Test
    void failsOnABucketErrorThatIsNotAConflict() {
        Storage mockStorage = mock(Storage.class);
        given(mockStorage.create(any(BucketInfo.class))).willThrow(new StorageException(403, "denied"));
        GcsClipStore mockStore = new GcsClipStore(mockStorage, new GcsStorageConfig(BUCKET), new ClipConfig(MAX_BYTES));

        assertThatThrownBy(mockStore::ensureBucket).isInstanceOf(ClipStorageException.class);
    }
}
