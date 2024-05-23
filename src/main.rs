use bevy::prelude::*;

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

    // Area light
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().normal(Dir3::X).size(1.0, 0.5)),
        material: standard_materials.add(StandardMaterial {
            emissive: Color::srgb(0.8, 1.2, 2.0),
            ..default()
        }),
        transform: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
        ..default()
    });

    ambient_light.brightness = 0.0;
}
