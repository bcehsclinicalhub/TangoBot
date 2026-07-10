<div class="hero-text-container">
  <img src="assets/hublogo.png" class="hero-logo-small" />
  <span class="hero-welcome">Welcome to Hub Wiki</span>
  <span class="hero-tagline">your clinical and operational wiki site for the Clinical Hub</span>
</div>

<div class="grid cards" markdown>

* :ambulance: **Paramedic Specialists**
    [Enter PS Page →](ps/index.md){ .md-button }

* <span style="color: #d32f2f;">☎</span> **Secondary Triage**
    [Enter STC Page →](stc/index.md){ .md-button }

</div>

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">

  <div id="shift-board" style="margin: 32px 0; max-width: 900px; width: 100%;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--md-typeset-a-color, #eaeaea); padding-bottom: 8px; margin-bottom: 16px; width: 100%;">
      <div style="display: flex; align-items: baseline; gap: 12px;">
        <h2 style="margin: 0; font-size: 1.4rem; font-weight: 700; border: none; padding: 0; color: var(--md-typeset-color, #333);">📅 EPOS Schedule</h2>
        <span id="sync-time" style="font-size: 0.85rem; color: #777; font-weight: 400;">Loading shifts...</span>
      </div>
    </div>
    
    <div style="border: 1px solid #e0e0e0; border-radius: 6px; width: 100%; box-sizing: border-box; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
      <table style="width: 100%; border-collapse: collapse; text-align: left; background-color: #fff; font-size: 0.9rem; margin: 0; table-layout: fixed;">
        <thead>
          <tr style="background-color: #fafafa; border-bottom: 1px solid #e0e0e0; color: #555;">
            <th style="padding: 12px; font-weight: 600; width: 28%;">Date</th>
            <th style="padding: 12px; font-weight: 600; width: 24%;">Shift Name</th>
            <th style="padding: 12px; font-weight: 600; width: 20%;">Hours</th>
            <th style="padding: 12px; font-weight: 600; width: 28%;">Physician</th>
          </tr>
        </thead>
        <tbody id="table-rows">
          <tr><td colspan="4" style="padding: 16px; text-align: center; color: #777;">Initializing view...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

## ❓ Need Help?
!!! caution "Support"
    This site is **NOT** supported by the BCEHS Help Desk | contact [Lee Roberts](mailto:lee.roberts@bcehs.ca) for feedback or support.

<script>
async function displaySchedule() {
  try {
    let response;
    const cacheBuster = "?t=" + Date.now();
    
    // Try data.xml first
    try {
      response = await fetch(window.location.origin + "/data.xml" + cacheBuster);
      if (!response.ok) throw new Error();
    } catch (e) {
      // Fallback to schedule.xml if data.xml fails
      response = await fetch(window.location.origin + "/schedule.xml" + cacheBuster);
      if (!response.ok) throw new Error("Could not load data.xml or schedule.xml");
    }

    const xmlText = await response.text();
    const parser = new DOMParser();
    const xml = parser.parseFromString(xmlText, "text/xml");
    if (xml.querySelector("parsererror")) { throw new Error("XML parsing error"); }

    const shiftLookup = {};
    xml.querySelectorAll("*|Shift").forEach(shift => {
      const id = shift.querySelector("*|ShiftId")?.textContent?.trim();
      const name = shift.querySelector("*|ShiftName")?.textContent?.trim();
      if (id) shiftLookup[id] = name;
    });

    const providerLookup = {};
    xml.querySelectorAll("*|Provider").forEach(provider => {
      const id = provider.querySelector("*|ProviderId")?.textContent?.trim();
      const name = provider.querySelector("*|PrintName")?.textContent?.trim() || provider.querySelector("*|ProviderName")?.textContent?.trim() || provider.querySelector("*|DisplayName")?.textContent?.trim() || provider.querySelector("*|Name")?.textContent?.trim();
      if (id) providerLookup[id] = name || "Unknown";
    });

    const today = new Date();
    const tomorrow = new Date();
    tomorrow.setDate(today.getDate() + 1);
    const todayStr = today.toLocaleDateString("en-CA");
    const tomorrowStr = tomorrow.toLocaleDateString("en-CA");

    const tbody = document.getElementById("table-rows");
    tbody.innerHTML = "";
    let rowsFound = 0;

    xml.querySelectorAll("*|Day").forEach(day => {
      const dayDate = day.querySelector("*|DayDate")?.textContent?.trim();
      if (dayDate !== todayStr && dayDate !== tomorrowStr) return;

      day.querySelectorAll("*|SchedShift").forEach(schedShift => {
        const shiftId = schedShift.querySelector("*|ShiftId")?.textContent?.trim();
        const shiftName = shiftLookup[shiftId] || shiftId || "Unknown Shift";

        schedShift.querySelectorAll("*|SchedProvider").forEach(sp => {
          const providerId = sp.querySelector("*|ProviderId")?.textContent?.trim();
          const providerName = providerLookup[providerId] || providerId || "Unknown Provider";

          const startIso = sp.querySelector("*|ScheduledStart")?.textContent;
          const endIso = sp.querySelector("*|ScheduledEnd")?.textContent;
          let hoursStr = "—";
          if (startIso && endIso) {
            const startTime = startIso.split("T")[1]?.substring(0, 5) || "";
            const endTime = endIso.split("T")[1]?.substring(0, 5) || "";
            hoursStr = startTime && endTime ? `${startTime} - ${endTime}` : "—";
          }

          let badge = '';
          if (dayDate === todayStr) {
            badge = `<span style="background-color: #e3f2fd; color: #0d47a1; padding: 3px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: 700; margin-right: 6px; inline-size: max-content;">TODAY</span>`;
          } else {
            badge = `<span style="background-color: #e8f5e9; color: #1b5e20; padding: 3px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: 700; margin-right: 6px; inline-size: max-content;">TOMORROW</span>`;
          }

          const row = document.createElement("tr");
          row.style.borderBottom = "1px solid #eaeaea";
          row.style.color = "var(--md-typeset-color, #222)";
          if (rowsFound % 2 === 1) { row.style.backgroundColor = "#fafafa"; }

          row.innerHTML = `
            <td style="padding: 10px 12px; display: flex; align-items: center; border: none; white-space: nowrap;"><div style="display: flex; align-items: center; min-width: max-content;">${badge} <span style="font-variant-numeric: tabular-nums;">${dayDate}</span></div></td>
            <td style="padding: 10px 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${shiftName}</td>
            <td style="padding: 10px 12px; font-variant-numeric: tabular-nums; white-space: nowrap;">${hoursStr}</td>
            <td style="padding: 10px 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">👤 ${providerName}</td>
          `;

          tbody.appendChild(row);
          rowsFound++;
        });
      });
    });

    if (rowsFound === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="padding: 24px; text-align: center; color: #777;">No assignments found for today or tomorrow.</td></tr>`;
    }

    const outputDate = xml.querySelector("*|DataOutputDate")?.textContent;
    if (outputDate) {
      document.getElementById("sync-time").innerText = "— Updated " + new Date(outputDate).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
  } catch (err) {
    console.error(err);
    document.getElementById("table-rows").innerHTML = `<tr><td colspan="4" style="padding: 16px; text-align: center; color: #d32f2f;">Error loading schedule.</td></tr>`;
  }
}

displaySchedule();
setInterval(displaySchedule, 3600000);
</script>