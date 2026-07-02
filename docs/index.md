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


## 📣 Wiki News
!!! success "Introducing Hub Wiki (May 2026)"
    **Clinical Hub Wiki** is the new home for information resources for both Paramedic Specialists and Secondary Triage Clinicians. Future state will see this site replace the Tango Drive and Resource pages from the MTS site. This wiki is designed as a one-stop resource for quick clinical and operational resources that can't be found on the Handbook or Intranet.

## 📅 Daily Operations Schedule
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


## ❓ Need Help?
!!! caution "Support"
    This site is **NOT** supported by the BCEHS Help Desk | contact [Lee Roberts](mailto:lee.roberts@bcehs.ca) for feedback or support.


<script>
  const fileUrl = 'https://raw.githubusercontent.com/bcehsclinicalhub/TangoBot/main/schedule.xml';

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