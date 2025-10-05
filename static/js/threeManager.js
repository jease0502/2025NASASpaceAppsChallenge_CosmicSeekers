import { DOMElements } from './dom.js';

const threeState = {
    scene: null, camera: null, renderer: null, clock: new THREE.Clock(),
    systemGroups: [],
    planetSystems: [], activeParticles: [],
    assets: { textures: {}, materials: {} }, composer: null, bloomPass: null,
    backgroundStars: [], isInitialized: false,
};

function createParticleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = 128;
    const context = canvas.getContext('2d');
    const gradient = context.createRadialGradient(canvas.width / 2, canvas.height / 2, 0, canvas.width / 2, canvas.height / 2, canvas.width / 2);
    gradient.addColorStop(0.0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.4, 'rgba(255,255,255,0.8)');
    gradient.addColorStop(1.0, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
    return new THREE.CanvasTexture(canvas);
}

async function loadAssets() {
    threeState.assets.textures.particle = createParticleTexture();
    const textureLoader = new THREE.TextureLoader();
    threeState.assets.materials.star = new THREE.MeshBasicMaterial({ color: 0xffff00, blending: THREE.AdditiveBlending });
    threeState.assets.materials.planet = new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.7, metalness: 0.1 });
    try {
        const texturePaths = {
            ice: "/static/textures/planet_ice.webp",
            earth: "/static/textures/planet_earthlike.webp",
            rocky: "/static/textures/planet_rocky.webp",
            lava: "/static/textures/planet_lava.webp",
            star_red_dwarf: "/static/textures/star_red_dwarf.webp",
            star_orange_dwarf: "/static/textures/star_orange_dwarf.webp",
            star_yellow_dwarf: "/static/textures/star_yellow_dwarf.webp",
            star_yellow_white: "/static/textures/star_yellow_white.webp",
            star_white: "/static/textures/star_white.webp",
            star_blue: "/static/textures/star_blue.webp",
        };
        const loadedTextures = await Promise.all(Object.values(texturePaths).map(path => textureLoader.loadAsync(path)));
        const textureKeys = Object.keys(texturePaths);
        for (let i = 0; i < textureKeys.length; i++) {
            const key = textureKeys[i];
            const texture = loadedTextures[i];
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.encoding = THREE.sRGBEncoding;
            threeState.assets.textures[key] = texture;
        }
        threeState.assets.materials.star.map = threeState.assets.textures.star_yellow_dwarf;
        threeState.assets.materials.planet.map = threeState.assets.textures.earth;
        threeState.assets.materials.star.needsUpdate = true;
        threeState.assets.materials.planet.needsUpdate = true;
    } catch (error) {
        console.error("❌ Error loading 3D resources.", error);
    }
}

function createBackgroundStars() {
    const createLayer = (count, speed, size, opacity) => {
        const vertices = [];
        for (let i = 0; i < count; i++) {
            vertices.push(THREE.MathUtils.randFloatSpread(4000), THREE.MathUtils.randFloatSpread(4000), THREE.MathUtils.randFloatSpread(4000));
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        const material = new THREE.PointsMaterial({ color: 0xffffff, size, transparent: true, opacity });
        const stars = new THREE.Points(geometry, material);
        stars.userData.rotationSpeed = speed;
        return stars;
    };
    [createLayer(5000, 0.0001, 0.8, 0.8), createLayer(5000, 0.00015, 0.6, 0.6), createLayer(5000, 0.0002, 0.4, 0.5)]
        .forEach(layer => {
            threeState.backgroundStars.push(layer);
            threeState.scene.add(layer);
        });
}
export const ThreeManager = {
    state: threeState,
    
    async init() {
        if (threeState.isInitialized) return;
        const { sceneContainer } = DOMElements;
        threeState.scene = new THREE.Scene();
        threeState.camera = new THREE.PerspectiveCamera(75, sceneContainer.clientWidth / sceneContainer.clientHeight, 0.1, 10000);
        threeState.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        threeState.renderer.setClearColor(0x000000, 0);
        threeState.renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
        sceneContainer.appendChild(threeState.renderer.domElement);
        new THREE.TextureLoader().load("/static/textures/background.webp", (texture) => { 
            texture.encoding = THREE.sRGBEncoding; // ✅ 保留正確顏色
            threeState.scene.background = texture; 
        });
        const renderPass = new THREE.RenderPass(threeState.scene, threeState.camera);
        threeState.bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(sceneContainer.clientWidth, sceneContainer.clientHeight), 1.8, 0.8, 0.6);
        threeState.composer = new THREE.EffectComposer(threeState.renderer);
        threeState.composer.addPass(renderPass);
        threeState.composer.addPass(threeState.bloomPass);
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
        threeState.scene.add(ambientLight);
        threeState.camera.position.set(0, 8, 20);
        threeState.camera.lookAt(threeState.scene.position);
        await loadAssets();
        createBackgroundStars();
        threeState.isInitialized = true;
    },

    clearScene() {
        threeState.systemGroups.forEach(group => threeState.scene.remove(group));
        threeState.systemGroups = [];
        threeState.planetSystems = [];
        threeState.activeParticles.forEach(p => threeState.scene.remove(p.points));
        threeState.activeParticles = [];
    },

    populateScene(systemsData, activeIndex = 0) {
        this.clearScene();
        const systemsArray = Array.isArray(systemsData) ? systemsData : [systemsData];
        
        const positions = [
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(-60, 30, -50),
            new THREE.Vector3(60, 50, -60),
        ];
        const positionMap = [activeIndex, (activeIndex + 1) % 3, (activeIndex + 2) % 3];
        
        systemsArray.forEach((systemData, i) => {
            const systemIndexInPositions = (systemsArray.length > 1) ? positionMap.indexOf(i) : 0;
            const systemGroup = new THREE.Group();
            systemGroup.name = `SystemGroup_${i}`;
            const { star, planets } = systemData;
            
            const stellarRadius = star.stellar_radius > 0 ? star.stellar_radius : 1.0;
            const starFinalScale = 2 * Math.pow(stellarRadius, 0.5);
            const starMaterial = threeState.assets.materials.star.clone();
            const starMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), starMaterial);
            starMesh.name = `Star_${i}`;
            starMesh.scale.setScalar(starFinalScale);
            systemGroup.add(starMesh);

            const pointLight = new THREE.PointLight(0xffffff, 2.5, 4000, 2);
            systemGroup.add(pointLight);
            
            const tempK = star.stellar_temp;
            let starColor = 0xffffcc; // Default color
            let starTexture = threeState.assets.textures.star_yellow_dwarf; // Default texture

            if (tempK < 3700) { starTexture = threeState.assets.textures.star_red_dwarf; starColor = 0xff9966; }
            else if (tempK < 5200) { starTexture = threeState.assets.textures.star_orange_dwarf; starColor = 0xffcc66; }
            else if (tempK < 6000) { starTexture = threeState.assets.textures.star_yellow_dwarf; starColor = 0xffffcc; }
            else if (tempK < 7500) { starTexture = threeState.assets.textures.star_yellow_white; starColor = 0xffffff; }
            else if (tempK < 10000) { starTexture = threeState.assets.textures.star_white; starColor = 0xccccff; }
            else { starTexture = threeState.assets.textures.star_blue; starColor = 0xaabbff; }
            
            starMaterial.map = starTexture;
            starMaterial.color.setHex(starColor);
            starMaterial.needsUpdate = true;
            pointLight.color.setHex(starColor);
            
            if (planets) {
                planets.forEach(data => {
                    const planetGroup = new THREE.Group();
                    planetGroup.name = `Planet_${data.planet_index}`; // <<< FIX: Added name to planet group
                    const orbitGroup = new THREE.Group();
                    const planetMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), threeState.assets.materials.planet.clone());
                    planetGroup.add(planetMesh);
                    orbitGroup.add(planetGroup);
                    
                    // --- Planet Scale and Orbit Calculation ---
                    const planetRadius = data.planet_radius > 0 ? data.planet_radius : 1.0;
                    const period = data.period > 0 ? data.period : 365.0;
                    const planetRadiusInSolarRadii = (planetRadius / 109.2);
                    // Fixed: Reduced scaling factor from 8 to 0.8 to make planets smaller than stars
                    const planetFinalScale = starFinalScale * Math.pow(planetRadiusInSolarRadii, 0.5) / Math.pow(stellarRadius, 0.5) * 0.8;
                    planetGroup.scale.setScalar(isNaN(planetFinalScale) ? 0.1 : planetFinalScale);
                    const baseOrbitRadius = starFinalScale * 4;
                    const orbitParams = { radius: baseOrbitRadius + 2.5 * Math.pow(period, 0.6), speed: (1 / Math.pow(period, 0.5)) / 8, start_angle: Math.random() * Math.PI * 2 };
                    const axial_rotation_speed = 0.01 / Math.log10(Math.max(0.1, data.planet_radius || 0.1) + 1);
                    
                    // --- Orbit Line ---
                    const points = [];
                    for (let i = 0; i <= 128; i++) {
                        const theta = (i / 128) * Math.PI * 2;
                        points.push(new THREE.Vector3(Math.cos(theta) * orbitParams.radius, 0, Math.sin(theta) * orbitParams.radius));
                    }
                    const orbitLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.2, transparent: true }));
                    orbitGroup.add(orbitLine);
                    
                    // --- Planet Texture Logic (MODIFIED for robustness) ---
                    const equilibriumTemp = data.equilibrium_temp;
                    let activeTexture = threeState.assets.textures.rocky; // Default fallback
                    if (equilibriumTemp) {
                        if (equilibriumTemp < 200) activeTexture = threeState.assets.textures.ice;
                        else if (equilibriumTemp < 400) activeTexture = threeState.assets.textures.earth;
                        else if (equilibriumTemp < 1000) activeTexture = threeState.assets.textures.rocky;
                        else activeTexture = threeState.assets.textures.lava;
                    }
                    planetMesh.material.map = activeTexture;
                    planetMesh.material.needsUpdate = true;

                    systemGroup.add(orbitGroup);
                    threeState.planetSystems.push({ planetGroup, planetMesh, orbitGroup, orbitParams, data, parentSystemGroup: systemGroup, axial_rotation_speed });
                });
            }
            
            systemGroup.position.copy(positions[systemIndexInPositions]);
            threeState.scene.add(systemGroup);
            threeState.systemGroups.push(systemGroup);
        });
    },
    
    explodePlanet(planetObject) {
        if (!planetObject) return;
        planetObject.visible = false;
        const particleCount = 2000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const planetWorldPosition = new THREE.Vector3();
        planetObject.getWorldPosition(planetWorldPosition);

        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            positions.set([planetWorldPosition.x, planetWorldPosition.y, planetWorldPosition.z], i3);
            const spherical = new THREE.Vector3(
                Math.random() * 2 - 1,
                Math.random() * 2 - 1,
                Math.random() * 2 - 1
            ).normalize().multiplyScalar(Math.random() * 0.5 + 0.1);
            velocities.set([spherical.x, spherical.y, spherical.z], i3);
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
        
        const material = new THREE.PointsMaterial({ size: 0.1, transparent: true, opacity: 1.0, blending: THREE.AdditiveBlending, depthWrite: false, map: threeState.assets.textures.particle });
        const points = new THREE.Points(geometry, material);
        points.userData.startTime = threeState.clock.getElapsedTime();
        threeState.scene.add(points);
        threeState.activeParticles.push({ points, duration: 2.5 });
    },
    updateAnimation(onUpdateCallback) {
        const elapsedTime = threeState.clock.getElapsedTime();
        threeState.planetSystems.forEach(system => {
            if (system.planetGroup.visible) {
                const { planetGroup, planetMesh, orbitParams, axial_rotation_speed } = system;
                planetGroup.position.x = Math.cos(elapsedTime * orbitParams.speed + orbitParams.start_angle) * orbitParams.radius;
                planetGroup.position.z = Math.sin(elapsedTime * orbitParams.speed + orbitParams.start_angle) * orbitParams.radius;
                planetMesh.rotation.y += axial_rotation_speed;
            }
        });

        threeState.systemGroups.forEach(group => {
            const star = group.children[0];
            if (star) { star.rotation.y += 0.0005; }
        });

        threeState.backgroundStars.forEach(layer => { layer.rotation.y += layer.userData.rotationSpeed; });
        
        for (let i = threeState.activeParticles.length - 1; i >= 0; i--) {
            const particleSystem = threeState.activeParticles[i];
            const { points, duration } = particleSystem;
            const timeAlive = threeState.clock.getElapsedTime() - points.userData.startTime;
            if (timeAlive > duration) {
                threeState.scene.remove(points);
                points.geometry.dispose();
                points.material.dispose();
                threeState.activeParticles.splice(i, 1);
                continue;
            }
            const positions = points.geometry.attributes.position.array;
            const velocities = points.geometry.attributes.velocity.array;
            for (let j = 0; j < positions.length; j+=3) {
                positions[j] += velocities[j];
                positions[j+1] += velocities[j+1];
                positions[j+2] += velocities[j+2];
            }
            points.geometry.attributes.position.needsUpdate = true;
            points.material.opacity = 1.0 - (timeAlive / duration);
        }

        if (onUpdateCallback) onUpdateCallback();
        if (threeState.composer) threeState.composer.render();
    },

    handleResize() {
        const { sceneContainer } = DOMElements;
        if (!threeState.isInitialized || !sceneContainer) return;
        const { clientWidth, clientHeight } = sceneContainer;
        if (clientWidth === 0 || clientHeight === 0) return;
        
        threeState.camera.aspect = clientWidth / clientHeight;
        threeState.camera.updateProjectionMatrix();
        threeState.renderer.setSize(clientWidth, clientHeight);
        threeState.composer.setSize(clientWidth, clientHeight);
    },
};