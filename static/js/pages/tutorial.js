import { ThreeManager } from '../threeManager.js';
import { DOMElements } from '../dom.js';

const showcaseItems = [
    { id: 'star-blue', title: 'O/B/A Type Star (Blue/White Star)', description: 'Surface temperature > 7,500K. These are the hottest and brightest stars in the universe, radiating dazzling blue-white light.', textureKey: 'star_blue', glow: { color: 0x8cb9ff, intensity: 2.5, size: 1.5 }, scale: 2 },
    { id: 'star-yellow', title: 'F/G Type Star (Yellow-White/Yellow Star)', description: 'Surface temperature 5,200K - 7,500K. Our Sun is a G-type star, emitting a warm yellow glow, making it an ideal cradle for life.', textureKey: 'star_yellow_dwarf', glow: { color: 0xffe8a1, intensity: 2.0, size: 1.2 }, scale: 2 },
    { id: 'star-orange', title: 'K Type Star (Orange Dwarf)', description: 'Surface temperature 3,700K - 5,200K. Slightly cooler and redder than the Sun, with extremely long lifespans, considered prime candidates for the search for alien life.', textureKey: 'star_orange_dwarf', glow: { color: 0xffb86c, intensity: 2.0, size: 1.2 }, scale: 2 },
    { id: 'star-red', title: 'M Type Star (Red Dwarf)', description: 'Surface temperature < 3,700K. The most common yet faintest stars in the universe, emitting dim red light with lifespans lasting trillions of years.', textureKey: 'star_red_dwarf', glow: { color: 0xff6c6c, intensity: 1.8, size: 1.1 }, scale: 2 },
    { id: 'planet-ice', title: 'Ice Giant Planet', description: 'Equilibrium temperature < 200K. A harsh world covered in ice and frozen gases, receiving only faint starlight.', textureKey: 'ice', glow: { color: 0xa1e8ff, intensity: 1.0, size: 1.0 }, scale: 1.8 },
    { id: 'planet-earth', title: 'Earth-like Planet', description: 'Equilibrium temperature 200K - 400K. Located in the habitable zone, with moderate temperatures, possibly supporting liquid water, atmosphere, and life.', textureKey: 'earth', glow: { color: 0x81c784, intensity: 1.2, size: 1.0 }, scale: 1.8 },
    { id: 'planet-rocky', title: 'Rocky Planet', description: 'Equilibrium temperature 400K - 1000K. Dry, hot rocky surfaces with little to no atmosphere, similar to Mercury or a barren Mars.', textureKey: 'rocky', glow: { color: 0xbcaaa4, intensity: 1.0, size: 1.0 }, scale: 1.8 },
    { id: 'planet-lava', title: 'Lava Planet', description: 'Equilibrium temperature > 1000K. Tidally locked and extremely close to its star, one side in eternal daylight, its surface covered by oceans of lava.', textureKey: 'lava', glow: { color: 0xffb74d, intensity: 1.5, size: 1.1 }, scale: 1.8 }
];

function createSceneInstance(container, config) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 4;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const assets = ThreeManager.state.assets;
    const geometry = new THREE.SphereGeometry(1, 16, 16);
    const texture = assets.textures[config.textureKey];
    if (texture) {
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    }
    const material = new THREE.MeshStandardMaterial({
        map: texture,
        emissive: new THREE.Color(config.glow.color),
        emissiveIntensity: config.glow.intensity * 0.1
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.setScalar(config.scale);
    scene.add(mesh);

    scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const pointLight = new THREE.PointLight(config.glow.color, config.glow.intensity * 2.0, 500);
    scene.add(pointLight);

    let isDragging = false, previousMouseX = 0;
    
    const onMouseDown = (e) => { isDragging = true; previousMouseX = e.clientX; };
    const onMouseUp = () => { isDragging = false; };
    const onMouseMove = (e) => {
        if (!isDragging) return;
        const deltaX = e.clientX - previousMouseX;
        mesh.rotation.y += deltaX * 0.01;
        previousMouseX = e.clientX;
    };
    
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    renderer.domElement.addEventListener('mouseleave', onMouseUp);
    renderer.domElement.addEventListener('mousemove', onMouseMove);

    const resizeObserver = new ResizeObserver(entries => {
        if (entries.length === 0 || !entries[0].contentRect) return;
        const { width, height } = entries[0].contentRect;
        if (width === 0 || height === 0) return;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });
    resizeObserver.observe(container);

    return {
        update: () => {
            if (renderer.getContext()) {
                renderer.render(scene, camera)
            }
        },
        cleanup: () => {
            resizeObserver.disconnect();
            
            renderer.domElement.removeEventListener('mousedown', onMouseDown);
            renderer.domElement.removeEventListener('mouseup', onMouseUp);
            renderer.domElement.removeEventListener('mouseleave', onMouseUp);
            renderer.domElement.removeEventListener('mousemove', onMouseMove);
            
            if (container && container.contains(renderer.domElement)) {
                 container.removeChild(renderer.domElement);
            }
            
            mesh.geometry.dispose();
            if (mesh.material.dispose) mesh.material.dispose();
            
            renderer.forceContextLoss();
            renderer.dispose();
        }
    };
}

export const Tutorial = {
    activeScenes: [],
    intersectionObservers: [],

    init() {
        const mainContainer = document.getElementById('celestial-showcase-container');
        if (!mainContainer) {
            console.error("Tutorial init failed: main container not found.");
            return;
        }
        
        this.cleanup();

        showcaseItems.forEach(item => {
            const itemWrapper = document.createElement('div');
            itemWrapper.className = 'showcase-item';

            const canvasContainer = document.createElement('div');
            canvasContainer.id = `canvas-container-${item.id}`;
            canvasContainer.className = 'tutorial-canvas-container';

            const descriptionDiv = document.createElement('div');
            descriptionDiv.className = 'showcase-description';
            descriptionDiv.innerHTML = `
                <h3 class="text-lg font-semibold text-gray-200">${item.title}</h3>
                <p class="text-sm text-gray-400">${item.description}</p>
            `;

            itemWrapper.appendChild(canvasContainer);
            itemWrapper.appendChild(descriptionDiv);
            mainContainer.appendChild(itemWrapper);

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const sceneInstance = createSceneInstance(canvasContainer, item);
                        this.activeScenes.push(sceneInstance);
                        observer.unobserve(entry.target);
                    }
                });
            }, { rootMargin: "200px" });
            
            observer.observe(canvasContainer);
            this.intersectionObservers.push(observer);
        });
    },

    update() {
        // --- START: MODIFICATION ---
        // 直接引用 Tutorial 物件，而不是 `this`
        if (Tutorial.activeScenes.length > 0) {
            Tutorial.activeScenes.forEach(scene => scene.update());
        }
        // --- END: MODIFICATION ---
    },

    cleanup() {
        console.log("Cleaning up Tutorial view...");
        
        this.activeScenes.forEach(scene => scene.cleanup());
        this.activeScenes = [];

        this.intersectionObservers.forEach(observer => observer.disconnect());
        this.intersectionObservers = [];
    }
};