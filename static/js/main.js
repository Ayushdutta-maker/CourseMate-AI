/**
 * CourseMate AI — Main Client Controller
 * Controls sidebar drawer, global history modal, document status synchronization,
 * and academic interactive behaviors.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Mobile Sidebar Controls
    const sidebar = document.getElementById('app-sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle-btn');
    const closeBtn = document.getElementById('sidebar-close-btn');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('sidebar-open');
        });
    }

    if (closeBtn && sidebar) {
        closeBtn.addEventListener('click', () => {
            sidebar.classList.remove('sidebar-open');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (sidebar && sidebar.classList.contains('sidebar-open')) {
            if (!sidebar.contains(e.target) && (!toggleBtn || !toggleBtn.contains(e.target))) {
                sidebar.classList.remove('sidebar-open');
            }
        }
    });

    // 2. Universal History Record Modal Viewer
    const modal = document.getElementById('universal-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalTypeBadge = document.getElementById('modal-type-badge');
    const modalDate = document.getElementById('modal-date');
    const modalBody = document.getElementById('modal-body-content');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    window.openHistoryModal = async function(itemId) {
        if (!modal) return;

        modal.classList.remove('hidden');
        modalBody.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-circle-notch fa-spin" style="font-size: 2rem; color: var(--gold-primary);"></i>
                <p style="margin-top: 1rem; color: var(--text-muted);">Loading stored academic record...</p>
            </div>
        `;

        try {
            const response = await fetch(`/api/history/${itemId}`);
            if (!response.ok) throw new Error("Record fetch failed");
            const data = await response.json();

            modalTitle.textContent = data.topic || "Academic Record";
            modalDate.textContent = `Recorded: ${data.timestamp || 'Archived Session'}`;
            modalTypeBadge.textContent = (data.type || 'NOTE').toUpperCase();

            // Set badge color class
            if (data.type === 'notes') {
                modalTypeBadge.className = 'badge badge-gold';
            } else if (data.type === 'quiz') {
                modalTypeBadge.className = 'badge badge-navy';
            } else if (data.type === 'chat') {
                modalTypeBadge.className = 'badge';
                modalTypeBadge.style.backgroundColor = 'var(--info-soft)';
                modalTypeBadge.style.color = 'var(--info)';
            } else {
                modalTypeBadge.className = 'badge badge-success';
            }

            // Render content based on type
            if (data.type === 'quiz') {
                let quizList = data.content;
                if (typeof quizList === 'string') {
                    try { quizList = JSON.parse(quizList); } catch(e) {}
                }
                
                if (Array.isArray(quizList)) {
                    let html = `<div style="display: flex; flex-direction: column; gap: 1rem;">`;
                    quizList.forEach((q, idx) => {
                        html += `
                            <div style="background-color: var(--surface-warm); padding: 1.25rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                                <h4 style="margin-bottom: 0.5rem; color: var(--text-navy);">0${idx + 1}. ${q.question}</h4>
                                <ul style="list-style: none; padding-left: 0; margin: 0.5rem 0;">
                                    ${(q.options || []).map(opt => `
                                        <li style="padding: 0.35rem 0.6rem; border-radius: 4px; font-size: 0.9rem; margin-bottom: 0.25rem; ${opt === q.answer ? 'background-color: var(--success-soft); color: var(--success); font-weight: 600;' : 'color: var(--text-slate);'}">
                                            ${opt === q.answer ? '✓ ' : '• '} ${opt}
                                        </li>
                                    `).join('')}
                                </ul>
                                ${q.explanation ? `<div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.5rem; border-top: 1px solid var(--border-subtle); padding-top: 0.35rem;"><em>Explanation:</em> ${q.explanation}</div>` : ''}
                            </div>
                        `;
                    });
                    html += `</div>`;
                    modalBody.innerHTML = html;
                } else {
                    modalBody.innerHTML = `<pre style="white-space: pre-wrap;">${JSON.stringify(quizList, null, 2)}</pre>`;
                }
            } else if (data.type === 'chat') {
                let chatObj = data.content;
                if (typeof chatObj === 'string') {
                    try { chatObj = JSON.parse(chatObj); } catch(e) {}
                }
                if (chatObj && chatObj.query && chatObj.response) {
                    modalBody.innerHTML = `
                        <div style="margin-bottom: 1.5rem;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: var(--gold-dark); text-transform: uppercase;">Student Inquiry</div>
                            <div style="background-color: var(--navy-primary); color: #FFF; padding: 1rem; border-radius: var(--radius-sm); margin-top: 0.35rem; font-size: 0.95rem;">
                                ${chatObj.query}
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; font-weight: 700; color: var(--gold-dark); text-transform: uppercase;">Assistant Response</div>
                            <div style="background-color: var(--surface-warm); padding: 1.25rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); margin-top: 0.35rem; line-height: 1.7;">
                                ${marked.parse(chatObj.response)}
                            </div>
                        </div>
                    `;
                } else {
                    modalBody.innerHTML = marked.parse(data.raw_content || "");
                }
            } else {
                // Notes or Career Roadmap (Markdown)
                modalBody.innerHTML = marked.parse(data.raw_content || data.content || "No content recorded.");
            }
        } catch (error) {
            modalBody.innerHTML = `<div style="color: var(--error); padding: 1rem;">Failed to load saved content.</div>`;
        }
    };

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        });
    }

    // 3. Document Status Synchronization for Active Corpus Badge
    async function syncDocumentStatus() {
        try {
            const res = await fetch('/api/document-status');
            if (res.ok) {
                const data = await res.json();
                const docTitleElem = document.getElementById('sidebar-doc-title');
                if (docTitleElem) {
                    if (data.has_document && data.filename) {
                        docTitleElem.textContent = data.filename;
                        docTitleElem.title = `${data.filename} (${data.chunk_count} chunks indexed)`;
                    } else {
                        docTitleElem.textContent = "General Academic Corpus";
                    }
                }
            }
        } catch (e) {
            // Silently fall back
        }
    }
    syncDocumentStatus();

    // 4. Smooth Anchor Scrolling for public pages
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId && targetId !== '#') {
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
});
