package com.bbfc.notification.in.rest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.service.DedupAlertService;

@WebMvcTest(IngestController.class)
class IngestControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private DedupAlertService dedupAlertService;

    @Test 
    void newEventReturns202Accepted() throws Exception {
        Alert alert = Alert.dispatch(
                new EventId("FE-1"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67)
        );
        given(dedupAlertService.handleAlert(any(), any(), any()))
                .willReturn(new DedupAlertService.Result(alert, true));

        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "FE-1",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.67
                                }
                                """))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("accepted"))
                .andExpect(jsonPath("$.eventId").value("FE-1"));
    }

    @Test
    void duplicateEventReturns200Ok() throws Exception {
        Alert alert = Alert.dispatch(
                new EventId("FE-1"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.67)
        );
        given(dedupAlertService.handleAlert(any(), any(), any()))
                .willReturn(new DedupAlertService.Result(alert, false));

        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "FE-1",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.67
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("duplicate"));
    }

    @Test
    void blankEventIdReturns400() throws Exception {
        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.67
                                }
                                """))
                .andExpect(status().isBadRequest());
    }
}
