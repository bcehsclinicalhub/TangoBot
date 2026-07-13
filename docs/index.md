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

// Cleaned up section header to only display the raw formatted date label
function renderDaySection(dateText, cards) {
    return `
    <section class="day-section">
        <div class="day-date" style="font-weight: 700; font-size: 0.95rem; color: var(--md-primary-fg-color); border-bottom: 1px solid var(--md-default-fg-color--lightest); padding-bottom: 0.25rem; margin-bottom: 0.5rem;">
            ${dateText}
        </div>
        ${cards.length ? cards.join("") : '<p style="color:var(--md-default-fg-color--light); padding-left:10px;">No shifts scheduled.</p>'}
    </section>`;
}

function renderDashboard(todayCards, tomorrowCards, todayStr, tomorrowStr) {
    const container = document.getElementById("schedule-content");
    if (!container) return;
    container.innerHTML = "";
    container.insertAdjacentHTML("beforeend", renderDaySection(todayStr, todayCards));
    container.insertAdjacentHTML("beforeend", renderDaySection(tomorrowStr, tomorrowCards));
}

async function displaySchedule() {
  const text = (parent, tag) => parent.querySelector(tag)?.textContent.trim() ?? "";

  try {
    const response = await fetch("assets/data.xml", {
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
      }
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const xmlText = await response.text();
    const xml = new DOMParser().parseFromString(xmlText, "application/xml");
    if (xml.querySelector("parsererror")) throw new Error("Invalid XML layout");

    const shiftLookup = {};
    xml.querySelectorAll("Shift").forEach(shift => {
      shiftLookup[text(shift, "ShiftId")] = text(shift, "ShiftName");
    });

    const providerLookup = {};
    xml.querySelectorAll("Provider").forEach(provider => {
      providerLookup[text(provider, "ProviderId")] =
          text(provider, "PrintName") ||
          text(provider, "ProviderName") ||
          text(provider, "DisplayName") ||
          "Unknown";
    });

    const todayObj = new Date();
    const tomorrowObj = new Date(todayObj);
    tomorrowObj.setDate(todayObj.getDate() + 1);

    const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const todayLabel = todayObj.toLocaleDateString('en-CA', dateOptions);
    const tomorrowLabel = tomorrowObj.toLocaleDateString('en-CA', dateOptions);

    const todayISO = todayObj.toISOString().slice(0, 10);
    const tomorrowISO = tomorrowObj.toISOString().slice(0, 10);

    const todayCards = [];
    const tomorrowCards = [];

    xml.querySelectorAll("Day").forEach(day => {
      const dayDate = text(day, "DayDate");
      if (dayDate !== todayISO && dayDate !== tomorrowISO) return;

      day.querySelectorAll("SchedShift").forEach(schedShift => {
        const shiftName = shiftLookup[text(schedShift, "ShiftId")] || "Duty Shift";

        schedShift.querySelectorAll("SchedProvider").forEach(provider => {
          const providerName = providerLookup[text(provider, "ProviderId")] || "Unknown Provider";

          const start = text(provider, "ScheduledStart").split("T")[1]?.substring(0, 5) ?? "00:00";
          const end = text(provider, "ScheduledEnd").split("T")[1]?.substring(0, 5) ?? "00:00";

          const cardHTML = renderShiftCard(shiftName, start, end, providerName);

          if (dayDate === todayISO) {
            todayCards.push(cardHTML);
          } else {
            tomorrowCards.push(cardHTML);
          }
        });
      });
    });

    const outputDate = xml.querySelector("DataOutputDate")?.textContent;
    if (outputDate) {
      document.getElementById("sync-time").textContent =
        "— Updated " + new Date(outputDate).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit"
        });
    }

    renderDashboard(todayCards, tomorrowCards, todayLabel, tomorrowLabel);

  } catch(err) {
    console.error(err);
    const container = document.getElementById("schedule-content");
    if (container) {
        container.innerHTML = `<div style="padding:20px;color:#d32f2f;text-align:center;font-weight:600;">⚠️ Error: ${err.message}</div>`;
    }
  }
}

displaySchedule();
setInterval(displaySchedule, 3600000);
</script>