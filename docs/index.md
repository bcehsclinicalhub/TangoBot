<div class="hero-text-container">
  <img src="assets/hublogo.png" class="hero-logo-small" />
  <span class="hero-welcome">Welcome to Hub Wiki</span>
  <span class="hero-tagline">your clinical and operational wiki site for the Clinical Hub</span>
</div>

<div class="grid cards" markdown>
* :ambulance: **Paramedic Specialists** [Enter PS Page →](ps/index.md){ .md-button }
* <span style="color: #d32f2f;">☎</span> **Secondary Triage** [Enter STC Page →](stc/index.md){ .md-button }
</div>

<div class="schedule-wrapper">
    <div id="shift-board" class="schedule-card">
        <div class="schedule-header">
            <div class="schedule-title">
                <h2>📅 EPOS Schedule</h2>
            </div>
            <div class="schedule-updated">
                <span id="sync-time">Loading...</span>
            </div>
        </div>
        <div class="schedule-body">
            <div id="schedule-content">
                <div class="schedule-loading">
                    <div class="loading-spinner"></div>
                    <p>Loading today's assignments...</p>
                </div>
            </div>
        </div>
    </div>
</div>

## ❓ Need Help?
!!! success "Support"
    This site is **NOT** supported by the BCEHS Help Desk | contact [Lee Roberts](mailto:lee.roberts@bcehs.ca) for any issues, feedback or support.

<script>
function getShiftPill(name) {
    const lower = name.toLowerCase();
    if (lower.startsWith("day")) return '<span class="shift-pill day">DAY</span>';
    if (lower.startsWith("swing")) return '<span class="shift-pill swing">SWING</span>';
    if (lower.startsWith("eve")) return '<span class="shift-pill evening">EVENING</span>';
    if (lower.startsWith("night")) return '<span class="shift-pill night">NIGHT</span>';
    return '<span class="shift-pill other">SHIFT</span>';
}

function renderShiftCard(shiftName, start, end, providerName) {
    return `
    <div class="shift-card">
        <div class="shift-main">
            ${getShiftPill(shiftName)}
            <div class="shift-code">${shiftName.replace(/^(Day|Swing|Eve|Night)\s*/i,"")}</div>
        </div>
        <div class="shift-time">
            <span class="time-pill">${start} – ${end}</span>
        </div>
        <div class="shift-provider">
            <span class="provider-chip">👤 ${providerName}</span>
        </div>
    </div>`;
}

function renderDaySection(title, icon, dateText, cards) {
    return `
    <section class="day-section">
        <h2 class="day-header">${icon} ${title}</h2>
        <div class="day-date">${dateText}</div>
        ${cards.length ? cards.join("") : '<p style="color:var(--md-default-fg-color--light); padding-left:10px;">No shifts scheduled.</p>'}
    </section>`;
}

function renderDashboard(todayCards, tomorrowCards, todayStr, tomorrowStr) {
    const container = document.getElementById("schedule-content");
    if (!container) return;
    container.innerHTML = "";
    container.insertAdjacentHTML("beforeend", renderDaySection("TODAY", "☀️", todayStr, todayCards));
    container.insertAdjacentHTML("beforeend", renderDaySection("TOMORROW", "🌙", tomorrowStr, tomorrowCards));
}

async function displaySchedule() {
    const text = (parent, tag) => parent.querySelector(tag)?.textContent.trim() ?? "";

    try {
        const response = await fetch("assets/data.xml", {
            cache: "no-store",
            headers: { "Cache-Control": "no-cache", "Pragma": "no-cache" }
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const xmlText = await response.text();
        const xml = new DOMParser().parseFromString(xmlText, "application/xml");
        if (xml.querySelector("parsererror")) throw new Error("Invalid XML layout format");

        // 1. Build Lookups
        const shiftLookup = {};
        xml.querySelectorAll("Shift").forEach(s => { shiftLookup[text(s, "ShiftId")] = text(s, "ShiftName"); });

        const providerLookup = {};
        xml.querySelectorAll("Provider").forEach(p => {
            providerLookup[text(p, "ProviderId")] = text(p, "PrintName") || text(p, "ProviderName") || "Unknown";
        });

        // 2. Normalize System Matching Dates
        const todayObj = new Date();
        const tomorrowObj = new Date();
        tomorrowObj.setDate(todayObj.getDate() + 1);

        const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        const todayStr = todayObj.toLocaleDateString('en-CA', dateOptions);
        const tomorrowStr = tomorrowObj.toLocaleDateString('en-CA', dateOptions);

        const toDateKey = (d) => d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
        const todayMatchKey = toDateKey(todayObj);
        const tomorrowMatchKey = toDateKey(tomorrowObj);

        // 3. Process Assignments
        const todayCards = [];
        const tomorrowCards = [];

        const allElements = xml.getElementsByTagName("*");
        const assignments = [];
        
        for (let el of allElements) {
            if (el.tagName !== "Shift" && el.querySelector("ShiftId") && el.querySelector("ProviderId")) {
                assignments.push(el);
            }
        }
        
        console.log(`Found ${assignments.length} structural shift row elements.`);

        assignments.forEach((row, index) => {
            // DIAGNOSTIC LOG: Let's inspect the very first row properties directly
            if (index === 0) {
                console.log("DIAGNOSTIC - First Row XML Tag:", row.tagName);
                console.log("DIAGNOSTIC - First Row Inner HTML:", row.innerHTML);
                // Check if date hides in an attribute
                console.log("DIAGNOSTIC - First Row Attributes:", Array.from(row.attributes).map(a => `${a.name}=${a.value}`));
            }

            const shiftId = text(row, "ShiftId");
            const providerId = text(row, "ProviderId");
            
            // Try explicit tags, fallback to common attribute keys if tags are empty
            let rawDate = text(row, "AssignmentDate") || 
                          text(row, "Date") || 
                          text(row, "ShiftDate") || 
                          row.getAttribute("AssignmentDate") || 
                          row.getAttribute("Date") || 
                          "";

            if (!rawDate) return;

            const normalizedDate = rawDate.replace(/\//g, '-');

            const shiftName = shiftLookup[shiftId] || "Duty Shift";
            const providerName = providerLookup[providerId] || "Unassigned";
            
            const startTime = text(row, "StartTime") || "00:00";
            const endTime = text(row, "EndTime") || "00:00";
            const cardHTML = renderShiftCard(shiftName, startTime, endTime, providerName);

            if (normalizedDate.includes(todayMatchKey)) {
                todayCards.push(cardHTML);
            } else if (normalizedDate.includes(tomorrowMatchKey)) {
                tomorrowCards.push(cardHTML);
            }
        });

        // 4. Update Interface Header
        const outputDate = xml.querySelector("DataOutputDate")?.textContent;
        if (outputDate) {
            document.getElementById("sync-time").textContent = "— Updated " + new Date(outputDate).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        } else {
            document.getElementById("sync-time").textContent = "— Live Sync Active";
        }

        renderDashboard(todayCards, tomorrowCards, todayStr, tomorrowStr);

    } catch (err) {
        console.error(err);
        const container = document.getElementById("schedule-content");
        if (container) {
            container.innerHTML = `<div style="padding:20px; color:#d32f2f; text-align:center; font-weight:600;">⚠️ Error parsing scheduling data: ${err.message}</div>`;
        }
    }
}

displaySchedule();
setInterval(displaySchedule, 300000);
</script>