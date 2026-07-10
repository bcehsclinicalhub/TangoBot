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

  <!-- Expanded max-width to 900px for a wider table display -->
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
    const fileUrl = window.location.origin + "/data.xml";
    const response = await fetch(fileUrl, {
      method: 'GET',
      headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
      cache: 'no-store'
    });
    
    if (!response.ok) { throw new Error(`HTTP Error: ${response.status}`); }
    const xmlText = await response.text();
    const parser = new DOMParser();
    const xml = parser.parseFromString(xmlText, "text/xml");
    if (xml.getElementsByTagName("parsererror").length > 0) { throw new Error("XML structure format error"); }

    const shiftLookup = {};
    Array.from(xml.getElementsByTagName("Shift")).forEach(shift => {
      const id = shift.getElementsByTagName("ShiftId")[0]?.textContent?.trim();
      const name = shift.getElementsByTagName("ShiftName")[0]?.textContent?.trim();
      if (id) shiftLookup[id] = name;
    });

    const providerLookup = {};
    Array.from(xml.getElementsByTagName("Provider")).forEach(provider => {
      const id = provider.getElementsByTagName("ProviderId")[0]?.textContent?.trim();
      const name = provider.getElementsByTagName("PrintName")[0]?.textContent?.trim() || 
                   provider.getElementsByTagName("ProviderName")[0]?.textContent?.trim() || 
                   provider.getElementsByTagName("DisplayName")[0]?.textContent?.trim() || 
                   provider.getElementsByTagName("Name")[0]?.textContent?.trim();
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

    Array.from(xml.getElementsByTagName("Day")).forEach(day => {
      const dayDate = day.getElementsByTagName("DayDate")[0]?.textContent?.trim();
      if (dayDate !== todayStr && dayDate !== tomorrowStr) return;

      Array.from(day.getElementsByTagName("SchedShift")).forEach(schedShift => {
        const shiftId = schedShift.getElementsByTagName("ShiftId")[0]?.textContent?.trim();
        const shiftName = shiftLookup[shiftId] || shiftId || "Unknown Shift";

        Array.from(schedShift.getElementsByTagName("SchedProvider")).forEach(sp => {
          const providerId = sp.getElementsByTagName("ProviderId")[0]?.textContent?.trim();
          const providerName = providerLookup[providerId] || providerId || "Unknown Provider";

          const startIso = sp.getElementsByTagName("ScheduledStart")[0]?.textContent;
          const endIso = sp.getElementsByTagName("ScheduledEnd")[0]?.textContent;
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

    const outputDate = xml.getElementsByTagName("DataOutputDate")[0]?.textContent;
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
