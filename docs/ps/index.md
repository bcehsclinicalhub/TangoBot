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

        //----------------------------------
        // Load XML (LIVE DATA)
        //----------------------------------

        const fileUrl = "/data.xml";

        const response = await fetch(fileUrl + "?t=" + Date.now());

        if (!response.ok) {
            throw new Error("Unable to load data.xml");
        }

        const xmlText = await response.text();

        const parser = new DOMParser();
        const xml = parser.parseFromString(xmlText, "text/xml");

        //----------------------------------
        // Safety check (invalid XML catch)
        //----------------------------------

        if (xml.querySelector("parsererror")) {
            throw new Error("XML parsing error");
        }

        //----------------------------------
        // Build Shift lookup table
        //----------------------------------

        const shiftLookup = {};

        xml.querySelectorAll("Shift").forEach(shift => {

            const id = shift.querySelector("ShiftId")?.textContent?.trim();
            const name = shift.querySelector("ShiftName")?.textContent?.trim();

            if (id) shiftLookup[id] = name;
        });

        //----------------------------------
        // Build Provider lookup table
        //----------------------------------

        const providerLookup = {};

        xml.querySelectorAll("Provider").forEach(provider => {

            const id = provider.querySelector("ProviderId")?.textContent?.trim();

            const name =
                provider.querySelector("PrintName")?.textContent?.trim() ||
                provider.querySelector("ProviderName")?.textContent?.trim() ||
                provider.querySelector("DisplayName")?.textContent?.trim() ||
                provider.querySelector("Name")?.textContent?.trim();

            if (id) providerLookup[id] = name || "Unknown";
        });

        //----------------------------------
        // Date handling (safe local format)
        //----------------------------------

        const today = new Date();
        const tomorrow = new Date();
        tomorrow.setDate(today.getDate() + 1);

        const todayStr = today.toLocaleDateString("en-CA");
        const tomorrowStr = tomorrow.toLocaleDateString("en-CA");

        //----------------------------------
        // Populate table
        //----------------------------------

        const tbody = document.getElementById("table-rows");
        tbody.innerHTML = "";

        let rowsFound = 0;

        xml.querySelectorAll("Day").forEach(day => {

            const dayDate = day.querySelector("DayDate")?.textContent?.trim();

            if (dayDate !== todayStr && dayDate !== tomorrowStr) return;

            day.querySelectorAll("SchedShift").forEach(schedShift => {

                const shiftId =
                    schedShift.querySelector("ShiftId")?.textContent?.trim();

                const shiftName = shiftLookup[shiftId] || shiftId || "Unknown Shift";

                schedShift.querySelectorAll("SchedProvider").forEach(sp => {

                    const providerId =
                        sp.querySelector("ProviderId")?.textContent?.trim();

                    const providerName =
                        providerLookup[providerId] || providerId || "Unknown Provider";

                    const row = document.createElement("tr");

                    row.innerHTML = `
                        <td style="padding:8px;border-top:1px solid #ddd;">${dayDate}</td>
                        <td style="padding:8px;border-top:1px solid #ddd;">${shiftName}</td>
                        <td style="padding:8px;border-top:1px solid #ddd;">${providerName}</td>
                    `;

                    tbody.appendChild(row);
                    rowsFound++;

                });

            });

        });

        //----------------------------------
        // No results fallback
        //----------------------------------

        if (rowsFound === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="3" style="padding:12px;text-align:center;">
                        No shifts found for today or tomorrow.
                    </td>
                </tr>`;
        }

        //----------------------------------
        // Sync timestamp (optional)
        //----------------------------------

        const outputDate = xml.querySelector("DataOutputDate")?.textContent;

        if (outputDate) {
            document.getElementById("sync-time").innerText =
                "Last checked: " + new Date(outputDate).toLocaleString();
        }

    } catch (err) {

        console.error(err);

        document.getElementById("table-rows").innerHTML = `
            <tr>
                <td colspan="3">Error loading schedule.</td>
            </tr>`;
    }
}

// Initial load
displaySchedule();

// Refresh every 5 minutes
setInterval(displaySchedule, 300000);
</script>