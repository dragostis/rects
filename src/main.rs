use bevy::{core_pipeline::fxaa::Fxaa, prelude::*};

mod plugins;

use plugins::{
    CameraController, CameraControllerPlugin, ScreenSpaceGlobalIlluminationBundle,
    ScreenSpaceGlobalIlluminationPlugin,
};

fn main() {
    App::new()
        .insert_resource(Msaa::Off)
        .add_plugins((
            DefaultPlugins,
            CameraControllerPlugin,
            ScreenSpaceGlobalIlluminationPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut ambient_light: ResMut<AmbientLight>,
) {
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            transform: Transform::from_translation(Vec3::new(1.8, 0.4, 0.0))
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        CameraController::default(),
        ScreenSpaceGlobalIlluminationBundle::default(),
        Fxaa::default(),
    ));

    // Surface
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(5.0, 5.0)),
        material: standard_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        }),
        ..default()
    });

    // Cube
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(0.1, 0.1, 0.1)),
        material: standard_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        }),
        transform: Transform::from_xyz(0.8, 0.15, 0.2),
        ..default()
    });

    // Sphere
    commands.spawn(PbrBundle {
        mesh: meshes.add(Sphere::new(0.1)),
        material: standard_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        }),
        transform: Transform::from_xyz(0.8, 0.15, -0.2),
        ..default()
    });

    // Area light door
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().normal(Dir3::X).size(1.0, 0.5)),
        material: standard_materials.add(StandardMaterial {
            emissive: Color::srgb(0.8 * 0.3, 1.2 * 0.3, 2.0 * 0.3),
            ..default()
        }),
        transform: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
        ..default()
    });

    // Area light floor
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(0.2, 0.2)),
        material: standard_materials.add(StandardMaterial {
            emissive: Color::srgb(0.2 * 0.8, 2.5 * 0.8, 0.4 * 0.8),
            ..default()
        }),
        transform: Transform::from_translation(Vec3::new(0.8, 0.001, 0.0)),
        ..default()
    });

    // Cube light
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(0.1, 0.1, 0.1)),
        material: standard_materials.add(StandardMaterial {
            emissive: Color::srgb(2.0 * 0.5, 1.2 * 0.5, 0.7 * 0.5),
            ..default()
        }),
        transform: Transform::from_xyz(1.2, 0.05, 0.8),
        ..default()
    });

    ambient_light.brightness = 0.0;
}
