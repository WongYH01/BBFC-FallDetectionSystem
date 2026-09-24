package com.bbfc.notification.out.storage;

import com.bbfc.notification.config.SupabaseStorageConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStorageException;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.web.client.RestClient;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.binaryEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalToJson;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.put;
import static com.github.tomakehurst.wiremock.client.WireMock.putRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SupabaseStorageClipStoreTest {

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance().build();

    private static final String OBJECT = "/object/skeleton-clips/room-12/FE-1.mp4";
    private static final String BUCKET = "/bucket";
    private static final byte[] CLIP_BYTES = "fake-mp4-bytes".getBytes(StandardCharsets.UTF_8);

    private SupabaseStorageClipStore store;

    @BeforeEach
    void setUp() {
        SupabaseStorageConfig config =
                new SupabaseStorageConfig(wireMock.baseUrl(), "test-key", "skeleton-clips");
        store = new SupabaseStorageClipStore(config, RestClient.builder());
    }

    private static ClipContent clip() {
        return new ClipContent("FE-1.mp4", "video/mp4", CLIP_BYTES.length,
                () -> new ByteArrayInputStream(CLIP_BYTES));
    }

    @Test
    void uploadsTheClipAndReturnsTheStorageKey() {
        wireMock.stubFor(put(urlEqualTo(OBJECT)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"Key\":\"skeleton-clips/room-12/FE-1.mp4\",\"Id\":\"abc\"}")));

        String key = store.store("room-12", new EventId("FE-1"), clip());

        assertThat(key).isEqualTo("skeleton-clips/room-12/FE-1.mp4");
        wireMock.verify(putRequestedFor(urlEqualTo(OBJECT))
                .withHeader("Authorization", equalTo("Bearer test-key"))
                .withHeader("Content-Type", equalTo("video/mp4"))
                .withHeader("Content-Length", equalTo(String.valueOf(CLIP_BYTES.length)))
                .withRequestBody(binaryEqualTo(CLIP_BYTES)));
    }

    @Test
    void wrapsAnUploadFailure() {
        wireMock.stubFor(put(urlEqualTo(OBJECT)).willReturn(aResponse().withStatus(500)));

        assertThatThrownBy(() -> store.store("room-12", new EventId("FE-1"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void failsWhenTheResponseCarriesNoKey() {
        wireMock.stubFor(put(urlEqualTo(OBJECT)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"Id\":\"abc\"}")));

        assertThatThrownBy(() -> store.store("room-12", new EventId("FE-1"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void rejectsARoomIdThatWouldEscapeTheBucket() {
        assertThatThrownBy(() -> store.store("../../etc", new EventId("FE-1"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void rejectsAnEventIdThatWouldEscapeTheBucket() {
        assertThatThrownBy(() -> store.store("room-12", new EventId("../../etc"), clip()))
                .isInstanceOf(ClipStorageException.class);
    }

    @Test
    void createsThePrivateBucketWithTheMp4Restriction() {
        wireMock.stubFor(post(urlEqualTo(BUCKET)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"name\":\"skeleton-clips\"}")));

        store.ensureBucket();

        wireMock.verify(postRequestedFor(urlEqualTo(BUCKET))
                .withHeader("Authorization", equalTo("Bearer test-key"))
                .withRequestBody(equalToJson("""
                        {"id":"skeleton-clips","name":"skeleton-clips","public":false,
                         "file_size_limit":8388608,"allowed_mime_types":["video/mp4"]}""")));
    }

    @Test
    void toleratesADuplicateBucket() {
        // storage-api answers a duplicate with 400, not 409 — the real code is in the body.
        wireMock.stubFor(post(urlEqualTo(BUCKET)).willReturn(aResponse()
                .withStatus(400)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"statusCode\":\"409\",\"error\":\"Duplicate\","
                        + "\"message\":\"The resource already exists\",\"code\":\"BucketAlreadyExists\"}")));

        assertThatCode(() -> store.ensureBucket()).doesNotThrowAnyException();
    }

    @Test
    void failsOnABucketErrorThatIsNotADuplicate() {
        wireMock.stubFor(post(urlEqualTo(BUCKET)).willReturn(aResponse()
                .withStatus(400)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"statusCode\":\"400\",\"error\":\"Error\",\"code\":\"InvalidRequest\"}")));

        assertThatThrownBy(() -> store.ensureBucket()).isInstanceOf(ClipStorageException.class);
    }
}
