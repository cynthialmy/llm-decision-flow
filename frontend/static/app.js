// API base URL
const API_BASE = 'http://localhost:8000';

// Analysis form handler
document.getElementById('analysisForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const transcript = document.getElementById('transcript').value;
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    submitBtn.disabled = true;
    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcript }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert('Error analyzing content: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        loading.style.display = 'none';
    }
});

function displayResults(data) {
    const result = document.getElementById('result');
    result.style.display = 'block';

    // Display decision
    const decisionDiv = document.getElementById('decision');
    const decisionText = document.getElementById('decisionText');
    const decisionRationale = document.getElementById('decisionRationale');

    decisionText.textContent = `Action: ${data.decision.action}`;
    decisionRationale.textContent = `Rationale: ${data.decision.rationale}`;

    if (data.decision.requires_human_review) {
        decisionDiv.classList.add('escalate');
        decisionText.innerHTML += ' <strong>(Requires Human Review)</strong>';
    } else if (data.decision.action === 'Allow') {
        decisionDiv.classList.add('allow');
    }

    // Display claims
    const claimsDiv = document.getElementById('claims');
    claimsDiv.innerHTML = data.claims.map(claim => `
        <div class="claim-item">
            <span class="badge domain">${claim.domain}</span>
            <strong>${claim.text}</strong>
            ${claim.is_explicit ? '' : '<span style="color: #666;">(implicit)</span>'}
        </div>
    `).join('');

    // Display risk assessment
    const riskDiv = document.getElementById('riskAssessment');
    const riskClass = `risk-${data.risk_assessment.tier.toLowerCase()}`;
    riskDiv.innerHTML = `
        <div class="claim-item">
            <span class="badge ${riskClass}">${data.risk_assessment.tier}</span>
            <p><strong>Reasoning:</strong> ${data.risk_assessment.reasoning}</p>
            <p><strong>Potential Harm:</strong> ${data.risk_assessment.potential_harm}</p>
            <p><strong>Estimated Exposure:</strong> ${data.risk_assessment.estimated_exposure}</p>
        </div>
    `;

    // Display evidence if available
    if (data.evidence) {
        document.getElementById('evidenceSection').style.display = 'block';
        const evidenceDiv = document.getElementById('evidence');
        let evidenceHtml = '';

        if (data.evidence.supporting.length > 0) {
            evidenceHtml += '<h4>Supporting Evidence:</h4>';
            evidenceHtml += data.evidence.supporting.map(item => `
                <div class="evidence-item">
                    <p>${item.text}</p>
                    <small>Source: ${item.source} | Quality: ${item.source_quality}</small>
                </div>
            `).join('');
        }

        if (data.evidence.contradicting.length > 0) {
            evidenceHtml += '<h4>Contradicting Evidence:</h4>';
            evidenceHtml += data.evidence.contradicting.map(item => `
                <div class="evidence-item">
                    <p>${item.text}</p>
                    <small>Source: ${item.source} | Quality: ${item.source_quality}</small>
                </div>
            `).join('');
        }

        evidenceDiv.innerHTML = evidenceHtml || '<p>No evidence retrieved.</p>';
    }

    // Display factuality assessments if available
    if (data.factuality_assessments && data.factuality_assessments.length > 0) {
        document.getElementById('factualitySection').style.display = 'block';
        const factualityDiv = document.getElementById('factuality');
        factualityDiv.innerHTML = data.factuality_assessments.map(fa => `
            <div class="claim-item">
                <p><strong>${fa.claim_text}</strong></p>
                <p>Status: ${fa.status} (confidence: ${(fa.confidence * 100).toFixed(1)}%)</p>
                <p>Reasoning: ${fa.reasoning}</p>
            </div>
        `).join('');
    }

    // Display policy interpretation if available
    if (data.policy_interpretation) {
        document.getElementById('policySection').style.display = 'block';
        const policyDiv = document.getElementById('policy');
        policyDiv.innerHTML = `
            <div class="claim-item">
                <p><strong>Violation:</strong> ${data.policy_interpretation.violation}</p>
                ${data.policy_interpretation.violation_type ? `<p><strong>Violation Type:</strong> ${data.policy_interpretation.violation_type}</p>` : ''}
                <p><strong>Policy Confidence:</strong> ${(data.policy_interpretation.policy_confidence * 100).toFixed(1)}%</p>
                <p><strong>Reasoning:</strong> ${data.policy_interpretation.reasoning}</p>
                ${data.policy_interpretation.allowed_contexts.length > 0 ? `<p><strong>Allowed Contexts:</strong> ${data.policy_interpretation.allowed_contexts.join(', ')}</p>` : ''}
            </div>
        `;
    }
}

// Review page functions
if (window.location.pathname.includes('review.html')) {
    loadReviews();
}

async function loadReviews() {
    try {
        const response = await fetch(`${API_BASE}/api/reviews`);
        const reviews = await response.json();

        const reviewsDiv = document.getElementById('reviews');
        if (reviews.length === 0) {
            reviewsDiv.innerHTML = '<p>No pending reviews.</p>';
            return;
        }

        reviewsDiv.innerHTML = reviews.map(review => `
            <div class="review-item">
                <h3>Review #${review.id}</h3>
                <p><strong>Transcript:</strong> ${review.transcript.substring(0, 200)}...</p>
                <p><strong>System Decision:</strong> ${review.system_decision.action}</p>
                <a href="review-detail.html?id=${review.id}">View Details</a>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading reviews:', error);
    }
}

// Metrics page functions
if (window.location.pathname.includes('dashboard.html')) {
    loadMetrics();
}

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics`);
        const metrics = await response.json();

        document.getElementById('highRiskExposure').textContent = (metrics.high_risk_exposure_rate * 100).toFixed(2) + '%';
        document.getElementById('overEnforcement').textContent = (metrics.over_enforcement_proxy * 100).toFixed(2) + '%';
        document.getElementById('disagreement').textContent = (metrics.model_human_disagreement * 100).toFixed(2) + '%';
        document.getElementById('reviewLoad').textContent = metrics.human_review_load;
        document.getElementById('avgTime').textContent = (metrics.avg_time_to_decision / 60).toFixed(1) + ' minutes';
        document.getElementById('totalDecisions').textContent = metrics.total_decisions;
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}
