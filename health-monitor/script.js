document.addEventListener('DOMContentLoaded', () => {
	const dataDisplay = document.querySelector('.data-display');
	const alertBox = document.querySelector('.alert');
	const callButton = document.querySelector('#call-doctor');
	const resetButton = document.querySelector('#reset');

	let intervalId;
	let abnormalDetected = false;
	let timeElapsed = 0; // Tracks elapsed time in seconds

	function fetchData() {
		if (abnormalDetected) return;

		const isFirst8Seconds = timeElapsed < 8;
		const isAbnormal = !isFirst8Seconds && Math.random() < 0.2; // 20% chance for abnormal after 8 seconds

		const data = {
			spo2: isAbnormal ? Math.floor(Math.random() * 5) + 90 : Math.floor(Math.random() * 10) + 95,
			heartRate: isAbnormal
				? Math.random() > 0.5
					? Math.floor(Math.random() * 20) + 40
					: Math.floor(Math.random() * 20) + 100
				: Math.floor(Math.random() * 40) + 60,
			bodyTemp: isAbnormal ? (Math.random() * 2 + 38).toFixed(1) : (Math.random() * 1.5 + 36).toFixed(1),
			ecg: isAbnormal ? 'Abnormal' : 'Normal',
		};

		dataDisplay.innerHTML = `
      <p><strong>SPO2:</strong> ${data.spo2}%</p>
      <p><strong>Heart Rate:</strong> ${data.heartRate} bpm</p>
      <p><strong>Body Temperature:</strong> ${data.bodyTemp}°C</p>
      <p><strong>ECG:</strong> ${data.ecg}</p>
    `;

		if (
			!isFirst8Seconds &&
			(data.spo2 < 95 || data.heartRate < 60 || data.heartRate > 100 || data.bodyTemp > 37.5 || data.ecg === 'Abnormal')
		) {
			abnormalDetected = true;
			alertBox.style.display = 'block';
			clearInterval(intervalId);
		}

		timeElapsed++;
	}

	function resetData() {
		abnormalDetected = false;
		alertBox.style.display = 'none';
		dataDisplay.innerHTML = `
      <p><strong>SPO2:</strong> --%</p>
      <p><strong>Heart Rate:</strong> -- bpm</p>
      <p><strong>Body Temperature:</strong> --°C</p>
      <p><strong>ECG:</strong> --</p>
    `;
		timeElapsed = 0; // Reset elapsed time
		intervalId = setInterval(fetchData, 1000);
	}

	intervalId = setInterval(fetchData, 1000);

	callButton.addEventListener('click', () => {
		window.location.href = 'tel:+1234567890'; // Replace with actual doctor's number
	});

	resetButton.addEventListener('click', resetData);
});
