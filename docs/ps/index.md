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

## 📅 Daily EPOS Schedule
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
  const fileUrl = '../data.xml';

  async function displaySchedule() {
    try {
      const response = await fetch(fileUrl + '?t=' + new Date().getTime());
      const textData = await response.text();
      
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(textData, "text/xml");

      const todayStr = new Date().toISOString().split('T')[0];
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      const tomorrowStr = tomorrow.toISOString().split('T')[0];

      const tbody = document.getElementById('table-rows');
      tbody.innerHTML = ''; 
      
      const assignments = xmlDoc.getElementsByTagName("Assignment"); 
      let foundAnyShifts = false;

      for (let i = 0; i < assignments.length; i++) {
        const shiftDate = assignments[i].getElementsByTagName("Date")[0]?.textContent;

        if (shiftDate === todayStr || shiftDate === tomorrowStr) {
          foundAnyShifts = true;
          const shiftName = assignments[i].getElementsByTagName("ShiftName")[0]?.textContent || "N/A";
          const providerName = assignments[i].getElementsByTagName("ProviderName")[0]?.textContent || "Vacant";

          const row = document.createElement('tr');
          row.style.borderBottom = "1px solid #eee";
          row.innerHTML = `
            <td style="padding: 10px;">${shiftDate}</td>
            <td style="padding: 10px;"><strong>${shiftName}</strong></td>
            <td style="padding: 10px;">${providerName}</td>
          `;
          tbody.appendChild(row);
        }
      }

      document.getElementById('sync-time').textContent = "Last checked: " + new Date().toLocaleTimeString();

      if (!foundAnyShifts) {
        tbody.innerHTML = '<tr><td colspan="3" style="padding:10px; text-align:center; color:#777;">No shifts found for today or tomorrow.</td></tr>';
      }

    } catch (error) {
      console.error("Oops:", error);
      document.getElementById('sync-time').textContent = "Error loading schedule data.";
    }
  }

  displaySchedule();
</script>
