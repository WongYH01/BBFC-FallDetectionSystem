package com.bbfc.notification.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import jakarta.persistence.LockModeType;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

public interface AlertJPARepository extends JpaRepository<AlertEntity,String> {
    @Query(value = """
        SELECT * FROM alerts.alerts
         WHERE state IN ('DISPATCHED','ESCALATING')
           AND next_escalation_at <= :now
         ORDER BY next_escalation_at
           FOR UPDATE SKIP LOCKED
         LIMIT :limit
        """, nativeQuery = true)
    List<AlertEntity> findDueForEscalation(@Param("now") Instant now, @Param("limit") int limit);

    // Derived query, not native: @Lock is ignored on native queries, so the lock has to ride on
    // a query Hibernate generates. Words between "find" and "By" are ignored by Spring Data.
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    Optional<AlertEntity> findForUpdateByEventId(String eventId);

}
