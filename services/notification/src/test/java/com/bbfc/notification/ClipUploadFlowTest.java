package com.bbfc.notification;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.SkeletonClip;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.out.persistence.AlertJPARepository;
import com.bbfc.notification.testsupport.AbstractIntegrationTest;
import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import java.nio.charset.StandardCharsets;

import static com.github.tomakehurst.wiremock.client.WireMock.aMultipart;
import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.binaryEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static com.github.tomakehurst.wiremock.client.WireMock.exactly;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.putRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * The clip goes to storage and to Telegram independently — these assert that neither failure
 * takes the other down with it.
 */
@AutoConfigureMockMvc
class ClipUploadFlowTest extends AbstractIntegrationTest {

    private static final String EVENT_ID = "FE-20260916-0077";
    private static final String SEND = "/bottest-token/sendMessage";
    private static final String ANIMATION = "/bottest-token/sendAnimation";
    private static final String OBJECT = "/object/skeleton-clips/room-12/" + EVENT_ID + ".mp4";
    private static final byte[] CLIP_BYTES = "fake-skeleton-mp4-bytes".getBytes(StandardCharsets.UTF_8);

    @RegisterExtension
    static WireMockExtension telegram = WireMockExtension.newInstance().build();

    @RegisterExtension
    static WireMockExtension storage = WireMockExtension.newInstance().build();

    @DynamicPropertySource
    static void baseUrls(DynamicPropertyRegistry registry) {
        registry.add("telegram.base-url", telegram::baseUrl);
        registry.add("supabase.storage.url", storage::baseUrl);
    }

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private AlertRepository alertRepository;

    @Autowired
    private AlertJPARepository alertJpaRepository;

    @BeforeEach
    void resetState() {
        // @SpringBootTest does not roll back, so state would leak into the next method.
        alertJpaRepository.deleteAll();
        telegram.resetAll();
        storage.resetAll();
        telegram.stubFor(WireMock.post(urlEqualTo(SEND)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":501}}")));
    }

    private void stubStorageOk() {
        storage.stubFor(WireMock.put(urlEqualTo(OBJECT)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"Key\":\"skeleton-clips/room-12/" + EVENT_ID + ".mp4\"}")));
    }

    private void stubTelegramClipOk() {
        telegram.stubFor(WireMock.post(urlEqualTo(ANIMATION)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":601,"
                        + "\"animation\":{\"file_id\":\"file-abc\"}}}")));
    }

    private void ingestEvent() throws Exception {
        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "%s",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.92
                                }
                                """.formatted(EVENT_ID)))
                .andExpect(status().isAccepted());
    }

    private static MockMultipartFile clipPart() {
        return new MockMultipartFile("file", "pose_skeleton.mp4", "video/mp4", CLIP_BYTES);
    }

    private org.springframework.test.web.servlet.ResultActions uploadClip() throws Exception {
        return mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", EVENT_ID).file(clipPart()));
    }

    @Test
    void clipIsStoredAndPostedAsAReplyToTheAlert() throws Exception {
        ingestEvent();
        stubStorageOk();
        stubTelegramClipOk();

        uploadClip()
                .andExpect(status().isAccepted())
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers
                        .jsonPath("$.sentToTelegram").value(true));

        // Both destinations got the same bytes — the guard on reading the upload twice.
        storage.verify(putRequestedFor(urlEqualTo(OBJECT))
                .withRequestBody(binaryEqualTo(CLIP_BYTES)));
        telegram.verify(postRequestedFor(urlEqualTo(ANIMATION))
                .withRequestBodyPart(aMultipart().withName("reply_to_message_id")
                        .withBody(equalTo("501")).build())
                .withRequestBodyPart(aMultipart().withName("animation")
                        .withBody(binaryEqualTo(CLIP_BYTES)).build()));

        SkeletonClip clip = reloadClip();
        assertThat(clip.storageKey()).isEqualTo("room-12/" + EVENT_ID + ".mp4");
        assertThat(clip.telegramFileId()).isEqualTo("file-abc");
        assertThat(clip.telegramMessageId()).isEqualTo(601L);
    }

    @Test
    void telegramFailureStillLeavesTheClipStoredAndReturns202() throws Exception {
        ingestEvent();
        stubStorageOk();
        telegram.stubFor(WireMock.post(urlEqualTo(ANIMATION)).willReturn(aResponse().withStatus(500)));

        uploadClip()
                .andExpect(status().isAccepted())
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers
                        .jsonPath("$.sentToTelegram").value(false));

        SkeletonClip clip = reloadClip();
        assertThat(clip.storedAt()).isNotNull();
        assertThat(clip.isDelivered()).isFalse();
    }

    @Test
    void storageFailureReturns500AndNeverReachesTelegram() throws Exception {
        ingestEvent();
        storage.stubFor(WireMock.put(urlEqualTo(OBJECT)).willReturn(aResponse().withStatus(500)));

        uploadClip().andExpect(status().isInternalServerError());

        telegram.verify(exactly(0), postRequestedFor(urlEqualTo(ANIMATION)));
        assertThat(reloadAlert().clip()).isEmpty();
    }

    @Test
    void aSecondClipIsRejectedWithoutUploadingAgain() throws Exception {
        ingestEvent();
        stubStorageOk();
        stubTelegramClipOk();
        uploadClip().andExpect(status().isAccepted());
        storage.resetRequests();

        uploadClip().andExpect(status().isConflict());

        storage.verify(exactly(0), putRequestedFor(urlEqualTo(OBJECT)));
    }

    @Test
    void aClipForAnUnknownEventIsRejected() throws Exception {
        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-does-not-exist").file(clipPart()))
                .andExpect(status().isNotFound());

        storage.verify(exactly(0), putRequestedFor(urlEqualTo(OBJECT)));
    }

    private Alert reloadAlert() {
        return alertRepository.findByEventId(new EventId(EVENT_ID)).orElseThrow();
    }

    private SkeletonClip reloadClip() {
        return reloadAlert().clip().orElseThrow();
    }
}
