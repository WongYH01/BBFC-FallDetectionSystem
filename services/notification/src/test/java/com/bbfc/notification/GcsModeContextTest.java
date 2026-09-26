package com.bbfc.notification;

import com.bbfc.notification.config.SupabaseStorageConfig;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.out.storage.GcsClipStore;
import com.bbfc.notification.out.storage.SupabaseStorageClipStore;
import com.bbfc.notification.testsupport.AbstractIntegrationTest;
import com.google.cloud.storage.Storage;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;

import static org.assertj.core.api.Assertions.assertThat;

class GcsModeContextTest extends AbstractIntegrationTest {

    @MockitoBean
    Storage storage;

    @Autowired
    ClipStore clipStore;

    @Autowired
    ApplicationContext context;

    @DynamicPropertySource
    static void gcsMode(DynamicPropertyRegistry registry) {
        registry.add("clips.store", () -> "gcs");
        registry.add("clips.gcs.bucket", () -> "test-clips");
    }

    @Test
    void wiresTheGcsStoreAndLeavesSupabaseOut() {
        assertThat(clipStore).isInstanceOf(GcsClipStore.class);
        assertThat(context.getBeansOfType(SupabaseStorageClipStore.class)).isEmpty();
        assertThat(context.getBeansOfType(SupabaseStorageConfig.class)).isEmpty();
    }
}
