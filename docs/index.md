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

  const text = (parent, tag) =>
    parent.querySelector(tag)?.textContent.trim() ?? "";

  try {

    const response = await fetch("assets/data.xml", {
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
      }
    });

    if (!response.ok)
      throw new Error(`HTTP ${response.status}`);

    const xmlText = await response.text();

    const xml = new DOMParser().parseFromString(xmlText, "application/xml");

    if (xml.querySelector("parsererror"))
      throw new Error("Invalid XML");

    // Build Shift lookup
    const shiftLookup = {};

    xml.querySelectorAll("Shift").forEach(shift => {
      shiftLookup[text(shift, "ShiftId")] = text(shift, "ShiftName");
    });

    // Build Provider lookup
    const providerLookup = {};

    xml.querySelectorAll("Provider").forEach(provider => {

      providerLookup[text(provider, "ProviderId")] =
          text(provider, "PrintName") ||
          text(provider, "ProviderName") ||
          text(provider, "DisplayName") ||
          text(provider, "Name") ||
          "Unknown";

    });

    // Today's dates
    const today = new Date();

    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    const todayStr = today.toISOString().slice(0,10);
    const tomorrowStr = tomorrow.toISOString().slice(0,10);

    console.log("Today:", todayStr);
    console.log("Tomorrow:", tomorrowStr);

    const tbody = document.getElementById("table-rows");
    tbody.innerHTML = "";

    let rows = 0;

    xml.querySelectorAll("Day").forEach(day => {

      const dayDate = text(day,"DayDate");

      if (dayDate !== todayStr && dayDate !== tomorrowStr)
        return;

      day.querySelectorAll("SchedShift").forEach(schedShift => {

        const shiftName =
          shiftLookup[text(schedShift,"ShiftId")] ||
          "Unknown Shift";

        schedShift.querySelectorAll("SchedProvider").forEach(provider => {

          const providerName =
            providerLookup[text(provider,"ProviderId")] ||
            "Unknown Provider";

          const start =
            text(provider,"ScheduledStart").split("T")[1]?.substring(0,5) ?? "";

          const end =
            text(provider,"ScheduledEnd").split("T")[1]?.substring(0,5) ?? "";

          const badge =
            dayDate === todayStr
              ? `<span style="background:#e3f2fd;color:#0d47a1;padding:3px 6px;border-radius:3px;font-size:.7rem;font-weight:bold;margin-right:6px;">TODAY</span>`
              : `<span style="background:#e8f5e9;color:#1b5e20;padding:3px 6px;border-radius:3px;font-size:.7rem;font-weight:bold;margin-right:6px;">TOMORROW</span>`;

          tbody.insertAdjacentHTML("beforeend",`
<tr style="border-bottom:1px solid #ececec">
<td style="padding:10px">${badge}${dayDate}</td>
<td style="padding:10px">${shiftName}</td>
<td style="padding:10px">${start} - ${end}</td>
<td style="padding:10px">👤 ${providerName}</td>
</tr>
`);

          rows++;

        });

      });

    });

    if (!rows) {

      tbody.innerHTML =
      `<tr>
          <td colspan="4"
              style="padding:20px;text-align:center;color:#777">
              No assignments found for today or tomorrow.
          </td>
      </tr>`;

    }

    const outputDate =
      xml.querySelector("DataOutputDate")?.textContent;

    if (outputDate) {

      document.getElementById("sync-time").textContent =
        "— Updated " +
        new Date(outputDate).toLocaleTimeString([],{
          hour:"2-digit",
          minute:"2-digit"
        });

    }

  }
  catch(err) {

    console.error(err);

    document.getElementById("table-rows").innerHTML =
    `<tr>
        <td colspan="4"
            style="padding:20px;color:#d32f2f;text-align:center;">
            ${err.message}
        </td>
    </tr>`;

  }

}

displaySchedule();

setInterval(displaySchedule, 3600000);
</script>
