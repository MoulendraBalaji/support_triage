document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.querySelector('.btn-primary');
    const statusTitle = document.querySelector('.status-title');
    const statusDetail = document.querySelector('.status-detail');
    const rewardHighlight = document.querySelector('.highlight');
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-header span:last-child');
    const scoreValue = document.querySelector('.score-value');
    const stepsValue = document.querySelector('.stat-item:nth-child(1) .stat-value');
    const remainingValue = document.querySelector('.stat-item:nth-child(2) .stat-value');
    const rewardValue = document.querySelector('.stat-item:nth-child(4) .stat-value');

    runBtn.addEventListener('click', () => {
        // Reset values for simulation
        statusTitle.textContent = 'Running Episode...';
        statusTitle.style.color = '#3b82f6';
        statusDetail.textContent = 'Executing support triage steps...';
        rewardHighlight.textContent = '0.000';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
        scoreValue.textContent = '0.000';
        stepsValue.textContent = '0';
        remainingValue.textContent = '5';
        rewardValue.textContent = '0.00';

        let step = 0;
        const totalSteps = 5;
        
        const interval = setInterval(() => {
            step++;
            const progress = (step / totalSteps) * 100;
            const reward = (step * 0.132).toFixed(3);
            
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}%`;
            stepsValue.textContent = step;
            remainingValue.textContent = totalSteps - step;
            rewardValue.textContent = (step * 0.068).toFixed(2);
            rewardHighlight.textContent = reward;

            if (step >= totalSteps) {
                clearInterval(interval);
                statusTitle.textContent = 'Episode Complete';
                statusTitle.style.color = '#10b981';
                statusDetail.textContent = `Processed ${totalSteps} issues in ${step} steps`;
                scoreValue.textContent = '0.400';
                
                // Add a small bounce effect to the score
                scoreValue.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    scoreValue.style.transform = 'scale(1)';
                }, 200);
            }
        }, 1000);
    });
});
