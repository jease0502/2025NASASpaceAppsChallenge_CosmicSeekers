// audioManager.js
// Manages all background music, including transitions and mute state.
const AudioState = {
    audio: new Audio(),
    currentTrack: null,
    targetVolume: 0.5,
    isMuted: false,
    fadeInterval: null,
    isUserInteracted: false,
};

const tracks = {
    start: '/static/music/start.mp3',
    planet_ice: '/static/music/planet_ice.mp3',
    planet_lava: '/static/music/planet_lava.mp3',
    planet_rocky: '/static/music/planet_rocky.mp3',
    planet_earthlike: '/static/music/planet_earthlike.mp3',
    kepler: '/static/music/kepler.mp3',
};

function fadeOut(onComplete) {
    if (AudioState.fadeInterval) clearInterval(AudioState.fadeInterval);
    if (AudioState.audio.volume === 0 || !AudioState.audio.src) {
        if (onComplete) onComplete();
        return;
    }
    AudioState.fadeInterval = setInterval(() => {
        const newVolume = AudioState.audio.volume - 0.05;
        if (newVolume <= 0) {
            AudioState.audio.volume = 0;
            AudioState.audio.pause();
            clearInterval(AudioState.fadeInterval);
            if (onComplete) onComplete();
        } else {
            AudioState.audio.volume = newVolume;
        }
    }, 50);
}

function fadeIn() {
    if (AudioState.fadeInterval) clearInterval(AudioState.fadeInterval);
    if (!AudioState.isUserInteracted) return;
    AudioState.audio.volume = 0;
    AudioState.audio.play().catch(e => console.error("Audio play failed:", e));
    AudioState.fadeInterval = setInterval(() => {
        const newVolume = AudioState.audio.volume + 0.05;
        if (newVolume >= AudioState.targetVolume) {
            AudioState.audio.volume = AudioState.targetVolume;
            clearInterval(AudioState.fadeInterval);
        } else {
            AudioState.audio.volume = newVolume;
        }
    }, 50);
}

export const AudioManager = {
    init() {
        const savedMuteState = localStorage.getItem('exoplanet_muted');
        this.setMuted(savedMuteState === 'true', false); // Do not update icon on init
        AudioState.audio.loop = true;
    },
    playMusic(trackName) {
        if (!tracks[trackName] || AudioState.currentTrack === trackName) {
            return;
        }
        AudioState.currentTrack = trackName;
        fadeOut(() => {
            AudioState.audio.src = tracks[trackName];
            fadeIn();
        });
    },
    setMuted(muted, updateIcon = true) {
        AudioState.isMuted = muted;
        AudioState.audio.muted = muted;
        localStorage.setItem('exoplanet_muted', muted);
        if (updateIcon) {
            const muteIcon = document.querySelector('#mute-icon');
            if (muteIcon) {
                muteIcon.textContent = muted ? 'volume_off' : 'volume_up';
            }
        }
    },
    userInteracted() {
        if (AudioState.isUserInteracted) return;
        console.log("User interaction detected, unlocking audio.");
        AudioState.isUserInteracted = true;
        if (AudioState.audio.src && AudioState.audio.paused) {
            fadeIn();
        }
    },
    // --- ★ 新增/修正: 補上 UI 需要的控制方法 ---
    toggleMute() {
        this.setMuted(!AudioState.isMuted, true);
    },
    isMuted() {
        return AudioState.isMuted;
    },
    getCurrentTrack() {
        return AudioState.currentTrack ? `${AudioState.currentTrack}.mp3` : '...';
    }
};