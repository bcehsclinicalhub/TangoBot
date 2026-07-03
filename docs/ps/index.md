<div class="hero-text-container">
  <img src="assets/hublogo.png" class="hero-logo-small" />
  <span class="hero-welcome">Hub Wiki | Paramedic Specialist</span>
</div>

---

## 🚑 Categories

| Section | Description | Link |
| :--- | :--- | :--- |
| **Clinical References** | Clinical guidelines and other clinical references | [Clinical References →](clinical/index.md) |
| **Operational Guidelines** | Practice Updates and operational resources | [Operational Guidelines →](operational/index.md) |
| **Safety Data Sheets** | Safety data sheets and other exposure resources | [Safety Data Sheets →](chemical/index.md) |

## 📅 EPOS Schedule
<div id="shift-board" style="margin: 20px 0; padding: 15px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #fafafa;">
  <p><small id="sync-time" style="color: #666;">Loading today and tomorrow's schedule...</small></p>
  <table style="width:100%; border-collapse: collapse; text-align: left; margin-top: 10px;">
    <thead>
      <tr style="background-color: #eee; border-bottom: 2px solid #ccc;">
        <th style="padding: 10px;">Date</th>
        <th style="padding: 10px;">Shift Name</th>
        <th style="padding: 10px;">Provider</th>
      </tr>
    </thead>
    <tbody id="table-rows">
      </tbody>
  </table>
</div>

## ⚠️ Critical Alerts
!!! danger "Fraser IFT IV Pump Trial"
    **July 2, 2026**: Starting this month (July 2026), BCEHS will begin piloting interfacility patient transports by PCPs using volumetric medication pumps to safely administer medications during transport. This will allow PCPs to transport patients who previously may have needed a nurse escort. [More Info on the Intranet](https://intranet.bcehs.ca/mlink/post/NDA3Ng){:target="_blank"}


## ⚕️ Clinical Updates
!!! warning "Latest Practice and CPG Updates"
    * **March 27, 2026 | Epi Infusion:** Updated epinephrine medication infusions to include 15 drop set change. [Visit the Handbook →](https://handbook.bcehs.ca){:target="_blank"}
    * **March 25, 2026 | Parenteral Ondansetron:** Now considered first-line parenteral antiemetic for most causes of nausea/vomiting. ODTs remain available. [More Info on Intranet →](https://intranet.bcehs.ca){:target="_blank"}
    * **March 20, 2026 | Updated M09:** Revised Neonatal Resuscitation flowchart now live. [Visit the Handbook →](https://handbook.bcehs.ca/clinical-practice-guidelines/m-pediatric-and-neonatal-emergencies/m09-neonatal-resuscitation/){:target="_blank"}

<script>
async function displaySchedule() {
  try {
    const fileUrl = window.location.origin + "/data.xml";
    const response = await fetch(fileUrl + "?t=" + Date.now());
    if (!response.ok) { throw new Error("Unable to load data.xml"); }
    const xmlText = await response.text();
    const parser = new DOMParser();
    const xml = parser.parseFromString(xmlText, "text/xml");
    if (xml.querySelector("parsererror")) { throw new Error("XML parsing error"); }

    const shiftLookup = {};
    xml.querySelectorAll("Shift").forEach(shift => {
      const id = shift.querySelector("ShiftId")?.textContent?.trim();
      const name = shift.querySelector("ShiftName")?.textContent?.trim();
      if (id) shiftLookup[id] = name;
    });

    const providerLookup = {};
    xml.querySelectorAll("Provider").forEach(provider => {
      const id = provider.querySelector("ProviderId")?.textContent?.trim();
      const name = provider.querySelector("PrintName")?.textContent?.trim() || provider.querySelector("ProviderName")?.textContent?.trim() || provider.querySelector("DisplayName")?.textContent?.trim() || provider.querySelector("Name")?.textContent?.trim();
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

    xml.querySelectorAll("Day").forEach(day => {
      const dayDate = day.querySelector("DayDate")?.textContent?.trim();
      if (dayDate !== todayStr && dayDate !== tomorrowStr) return;

      day.querySelectorAll("SchedShift").forEach(schedShift => {
        const shiftId = schedShift.querySelector("ShiftId")?.textContent?.trim();
        const shiftName = shiftLookup[shiftId] || shiftId || "Unknown Shift";

        schedShift.querySelectorAll("SchedProvider").forEach(sp => {
          const providerId = sp.querySelector("ProviderId")?.textContent?.trim();
          const providerName = providerLookup[providerId] || providerId || "Unknown Provider";

          let badge = '';
          if (dayDate === todayStr) {
            badge = `<span style="background-color: #e3f2fd; color: #0d47a1; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">TODAY</span>`;
          } else {
            badge = `<span style="background-color: #e8f5e9; color: #1b5e20; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">TOMORROW</span>`;
          }

          const row = document.createElement("tr");
          row.style.borderBottom = "1px solid #e0e0e0";
          if (rowsFound % 2 === 1) { row.style.backgroundColor = "#f9f9f9"; }

          row.innerHTML = `
            <td style="padding: 12px 16px; font-size: 0.9rem; color: #333; display: flex; align-items: center; border: none;">${badge} <span style="font-family: monospace;">${dayDate}</span></td>
            <td style="padding: 12px 16px; font-size: 0.9rem; color: #000;"><strong style="color: #2c3e50;">${shiftName}</strong></td>
            <td style="padding: 12px 16px; font-size: 0.9rem; color: #2c3e50; font-weight: 500;">👤 ${providerName}</td>
          `;

          tbody.appendChild(row);
          rowsFound++;
        });
      });
    });

    if (rowsFound === 0) {
      tbody.innerHTML = `<tr><td colspan="3" style="padding: 30px; text-align: center; color: #777; font-size: 0.9rem;">No assignments found for today or tomorrow.</td></tr>`;
    }

    const outputDate = xml.querySelector("DataOutputDate")?.textContent;
    if (outputDate) {
      document.getElementById("sync-time").innerText = "Updated: " + new Date(outputDate).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
  } catch (err) {
    console.error(err);
    document.getElementById("table-rows").innerHTML = `<tr><td colspan="3" style="padding: 20px; text-align: center; color: #d32f2f;">Error loading schedule.</td></tr>`;
  }
}

displaySchedule();
setInterval(displaySchedule, 3600000); // Corrected to 1 hour (3,600,000 ms)
</script>