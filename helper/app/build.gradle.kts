plugins {
    id("com.android.application")
}

android {
    namespace = "dev.aua.helper"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.aua.helper"
        // Deliberately low: the helper must install on whatever the fleet already runs.
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
    }

    buildFeatures {
        // HelperService reports BuildConfig.VERSION_NAME in the `helper.info` handshake.
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        // The host installs this over adb on a dev device; a debug build keeps the
        // signing story trivial and marks the package DEBUGGABLE for run-as access.
        getByName("debug") {
            isMinifyEnabled = false
        }
    }
}
