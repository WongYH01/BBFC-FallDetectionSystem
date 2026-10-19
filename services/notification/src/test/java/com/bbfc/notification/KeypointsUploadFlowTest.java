package com.bbfc.notification;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.service.TriageAlertService;
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
import org.springframework.test.web.servlet.ResultActions;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.function.BooleanSupplier;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.equalToJson;
import static com.github.tomakehurst.wiremock.client.WireMock.exactly;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.putRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.urlMatching;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** Upload and triage can land in either order; the file must end up under its label both ways. */
@AutoConfigureMockMvc
class KeypointsUploadFlowTest extends AbstractIntegrationTest {

    private static final String EVENT_ID = "FE-20260916-0088";
    private static final String INCOMING = "keypoints/incoming/room-12/" + EVENT_ID + ".npz";
    private static final String GENUINE = "keypoints/labelled/genuine/room-12/" + EVENT_ID + ".npz";
    private static final String MOVE = "/object/move";
    private static final byte[] NPZ = "PK\u0003\u0004fake-npz".getBytes(StandardCharsets.ISO_8859_1);

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

    @Autowired
    private TriageAlertService triageAlertService;

    @BeforeEach
    void resetState() {
        alertJpaRepository.deleteAll();
        telegram.resetAll();
        storage.resetAll();
        telegram.stubFor(WireMock.post(urlMatching("/bottest-token/.*")).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":501}}")));
        storage.stubFor(WireMock.put(urlMatching("/object/skeleton-clips/keypoints/.*")).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"Key\":\"ignored\"}")));
    }

    private void stubMove(int status) {
        storage.stubFor(WireMock.post(urlEqualTo(MOVE)).willReturn(aResponse()
                .withStatus(status)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"message\":\"moved\"}")));
    }

    private void ingestAndAcknowledge() throws Exception {
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
        Alert alert = reloadAlert();
        alert.acknowledge("@nurse_lim", Instant.now());
        alertRepository.save(alert);
    }

    private ResultActions upload() throws Exception {
        MockMultipartFile file = new MockMultipartFile("file", "fall.npz", "application/octet-stream", NPZ);
        return mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/keypoints", EVENT_ID).file(file));
    }

    private void triage() {
        triageAlertService.triage("cbq-1", new EventId(EVENT_ID), Outcome.GENUINE, Instant.now());
    }

    @Test
    void uploadThenTriageMovesTheFileUnderItsLabel() throws Exception {
        ingestAndAcknowledge();
        stubMove(200);
        upload().andExpect(status().isAccepted());
        assertThat(keypointsKey()).isEqualTo(INCOMING);

        triage();

        // The move runs after commit on another thread.
        waitUntil(() -> GENUINE.equals(keypointsKey()));
        storage.verify(postRequestedFor(urlEqualTo(MOVE)).withRequestBody(equalToJson("""
                {"bucketId":"skeleton-clips","sourceKey":"%s","destinationKey":"%s"}"""
                .formatted(INCOMING, GENUINE))));
    }

    @Test
    void triageThenUploadStoresStraightUnderTheLabel() throws Exception {
        ingestAndAcknowledge();
        triage();

        upload().andExpect(status().isAccepted());

        assertThat(keypointsKey()).isEqualTo(GENUINE);
        storage.verify(putRequestedFor(urlEqualTo("/object/skeleton-clips/" + GENUINE)));
        storage.verify(exactly(0), postRequestedFor(urlEqualTo(MOVE)));
    }

    @Test
    void aSecondUploadIsRejected() throws Exception {
        ingestAndAcknowledge();
        upload().andExpect(status().isAccepted());

        upload().andExpect(status().isConflict());
    }

    @Test
    void aFailedMoveLeavesTheTriageAndTheIncomingKeyInPlace() throws Exception {
        ingestAndAcknowledge();
        stubMove(500);
        upload().andExpect(status().isAccepted());

        triage();

        waitUntil(() -> !storage.findAll(postRequestedFor(urlEqualTo(MOVE))).isEmpty());
        Alert alert = reloadAlert();
        assertThat(alert.state()).isEqualTo(AlertState.TRIAGED);
        assertThat(alert.keypoints().orElseThrow().storageKey()).isEqualTo(INCOMING);
    }

    private Alert reloadAlert() {
        return alertRepository.findByEventId(new EventId(EVENT_ID)).orElseThrow();
    }

    private String keypointsKey() {
        return reloadAlert().keypoints().map(k -> k.storageKey()).orElse(null);
    }

    private static void waitUntil(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 5_000;
        while (!condition.getAsBoolean()) {
            assertThat(System.currentTimeMillis()).as("condition not met within 5 s").isLessThan(deadline);
            Thread.sleep(50);
        }
    }
}
