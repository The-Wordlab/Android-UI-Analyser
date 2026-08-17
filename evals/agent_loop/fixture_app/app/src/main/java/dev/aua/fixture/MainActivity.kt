package dev.aua.fixture

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button as ComposeButton
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    private data class Product(val slug: String, val name: String, val cents: Int)

    private enum class SortMode { NAME, PRICE }

    private val products = listOf(
        Product("alpine_mug", "Alpine Mug", 799),
        Product("beacon_lamp", "Beacon Lamp", 1250),
        Product("cedar_notebook", "Cedar Notebook", 425),
        Product("drift_puzzle", "Drift Puzzle", 1800),
    )
    private val handler = Handler(Looper.getMainLooper())
    private val preferences by lazy { getSharedPreferences("fixture", MODE_PRIVATE) }
    private var screenGeneration = 0
    private var onHome = true
    private var lastNotificationResult: Boolean? = null
    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        lastNotificationResult = granted
        showNotificationPermission()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (onHome) {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                } else {
                    showHome()
                }
            }
        })
        showHome()
    }

    private fun showHome() {
        screenGeneration += 1
        onHome = true
        val content = verticalContainer()
        content.id = R.id.fixture_home
        content.addView(title("AUA Agent Loop Fixture"))
        content.addView(body("Choose a fictional, deterministic testing lane."))
        content.addView(menuButton("Classic View grid", R.id.open_classic_grid) { showClassicGrid() })
        content.addView(menuButton("Compose grid", R.id.open_compose_grid) { showComposeGrid() })
        content.addView(menuButton("Async recovery", R.id.open_async_recovery) { showAsyncRecovery() })
        content.addView(menuButton("Canvas lab", R.id.open_canvas_lab) { showCanvasLab() })
        content.addView(
            menuButton("Notification permission", R.id.open_notification_permission) {
                showNotificationPermission()
            },
        )
        content.addView(menuButton("Reset fixture", R.id.fixture_reset) { resetFixture() })
        setContentView(scroll(content))
    }

    private fun resetFixture() {
        preferences.edit().clear().apply()
        lastNotificationResult = null
        screenGeneration += 1
        showHome()
    }

    private fun showClassicGrid() {
        onHome = false
        val content = screenContainer("Classic View grid")
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        controls.addView(
            menuButton("Sort by name", R.id.classic_sort_name) {
                preferences.edit().putString("classic_sort", SortMode.NAME.name).apply()
                showClassicGrid()
            },
            weightedParams(),
        )
        controls.addView(
            menuButton("Sort by price", R.id.classic_sort_price) {
                preferences.edit().putString("classic_sort", SortMode.PRICE.name).apply()
                showClassicGrid()
            },
            weightedParams(),
        )
        content.addView(controls)

        val mode = savedSort("classic_sort")
        content.addView(body("Current order: ${mode.name.lowercase()}"))
        val grid = LinearLayout(this).apply {
            id = R.id.classic_grid
            orientation = LinearLayout.VERTICAL
            contentDescription = "Classic products ordered by ${mode.name.lowercase()}"
        }
        sortedProducts(mode).forEachIndexed { index, product ->
            grid.addView(classicProductCard(product, index))
        }
        content.addView(grid)
        setContentView(scroll(content))
    }

    private fun classicProductCard(product: Product, index: Int): View {
        return LinearLayout(this).apply {
            id = R.id.classic_item_card
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(18), dp(16), dp(18), dp(16))
            setBackgroundColor(if (index % 2 == 0) Color.WHITE else Color.rgb(239, 241, 252))
            contentDescription = "${product.name}, ${price(product.cents)}, position ${index + 1}"

            addView(TextView(context).apply {
                id = R.id.classic_item_name
                text = product.name
                textSize = 18f
                setTextColor(Color.rgb(32, 33, 36))
            }, weightedParams())
            addView(TextView(context).apply {
                id = R.id.classic_item_price
                text = price(product.cents)
                textSize = 18f
                gravity = Gravity.END
                setTextColor(Color.rgb(32, 33, 36))
            }, weightedParams())
        }
    }

    private fun showComposeGrid() {
        onHome = false
        setContent {
            var mode by remember { mutableStateOf(savedSort("compose_sort")) }
            val ordered = sortedProducts(mode)
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .semantics { testTagsAsResourceId = true }
                            .testTag("compose_grid_screen")
                            .padding(16.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text("Compose grid", style = MaterialTheme.typography.headlineMedium)
                            ComposeButton(
                                onClick = { resetFixture() },
                                modifier = Modifier.testTag("fixture_reset"),
                            ) { Text("Reset fixture") }
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            ComposeButton(
                                onClick = {
                                    mode = SortMode.NAME
                                    preferences.edit().putString("compose_sort", mode.name).apply()
                                },
                                modifier = Modifier.weight(1f).testTag("compose_sort_name"),
                            ) { Text("Sort by name") }
                            ComposeButton(
                                onClick = {
                                    mode = SortMode.PRICE
                                    preferences.edit().putString("compose_sort", mode.name).apply()
                                },
                                modifier = Modifier.weight(1f).testTag("compose_sort_price"),
                            ) { Text("Sort by price") }
                        }
                        Text(
                            "Current order: ${mode.name.lowercase()}",
                            modifier = Modifier.testTag("compose_order_status"),
                        )
                        LazyColumn(
                            modifier = Modifier.fillMaxWidth().testTag("compose_grid"),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            items(ordered, key = { it.slug }) { product ->
                                Card(
                                    modifier = Modifier.fillMaxWidth().testTag("compose_item_${product.slug}"),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth().padding(18.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                    ) {
                                        Text(product.name, modifier = Modifier.testTag("compose_name_${product.slug}"))
                                        Text(price(product.cents), modifier = Modifier.testTag("compose_price_${product.slug}"))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun showAsyncRecovery() {
        onHome = false
        val content = screenContainer("Async recovery")
        val generation = screenGeneration
        val progress = ProgressBar(this).apply { contentDescription = "Loading expedition status" }
        val status = body("Loading expedition status…").apply {
            id = R.id.async_status
            contentDescription = "Loading expedition status"
        }
        content.addView(progress)
        content.addView(status)
        setContentView(scroll(content))

        handler.postDelayed({
            if (generation != screenGeneration) return@postDelayed
            content.removeView(progress)
            status.text = "Temporary signal error"
            status.contentDescription = "Temporary signal error"
            content.addView(menuButton("Retry", R.id.async_retry) {
                showAsyncSuccessAttempt(content, status)
            })
        }, ASYNC_DELAY_MS)
    }

    private fun showAsyncSuccessAttempt(content: LinearLayout, status: TextView) {
        screenGeneration += 1
        val generation = screenGeneration
        content.findViewById<View?>(R.id.async_retry)?.let(content::removeView)
        status.text = "Loading expedition status…"
        status.contentDescription = "Loading expedition status"
        val progress = ProgressBar(this).apply { contentDescription = "Retry in progress" }
        content.addView(progress, 2)
        handler.postDelayed({
            if (generation != screenGeneration) return@postDelayed
            content.removeView(progress)
            status.text = "Expedition ready: Aurora Trail"
            status.contentDescription = "Expedition ready, Aurora Trail"
        }, ASYNC_DELAY_MS)
    }

    @Suppress("SetJavaScriptEnabled")
    private fun showCanvasLab() {
        onHome = false
        val content = screenContainer("Local WebView canvas")
        val webView = WebView(this).apply {
            id = R.id.canvas_webview
            contentDescription = "Canvas lab WebView"
            settings.javaScriptEnabled = true
            settings.allowFileAccess = true
            loadUrl("file:///android_asset/canvas_lab.html")
        }
        content.addView(webView, LinearLayout.LayoutParams(MATCH_PARENT, dp(430)))
        setContentView(content)
    }

    private fun showNotificationPermission() {
        onHome = false
        val content = screenContainer("Notification permission")
        content.addView(body("Request the real Android notification permission dialog."))
        content.addView(menuButton("Request notifications", R.id.notification_request) {
            if (Build.VERSION.SDK_INT >= 33) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        })
        val status = when {
            Build.VERSION.SDK_INT < 33 -> "Notifications do not require runtime permission on this API."
            lastNotificationResult == true -> "Last request result: notification permission granted."
            lastNotificationResult == false -> "Last request result: notification permission denied."
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED ->
                "Notification permission is granted."
            else -> "Notification permission is not granted."
        }
        content.addView(body(status))
        setContentView(scroll(content))
    }

    private fun screenContainer(name: String): LinearLayout {
        screenGeneration += 1
        return verticalContainer().apply {
            val header = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            header.addView(title(name), weightedParams())
            header.addView(menuButton("Reset fixture", R.id.fixture_reset) { resetFixture() })
            addView(header)
        }
    }

    private fun verticalContainer() = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(18), dp(20), dp(18), dp(24))
        setBackgroundColor(Color.rgb(247, 247, 252))
    }

    private fun scroll(child: View) = ScrollView(this).apply {
        isFillViewport = true
        addView(child, ViewGroup.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
    }

    private fun title(value: String) = TextView(this).apply {
        text = value
        textSize = 28f
        setTextColor(Color.rgb(32, 33, 36))
        setPadding(0, dp(10), 0, dp(10))
    }

    private fun body(value: String) = TextView(this).apply {
        text = value
        textSize = 17f
        setTextColor(Color.rgb(60, 64, 67))
        setPadding(0, dp(12), 0, dp(12))
    }

    private fun menuButton(label: String, idValue: Int, action: () -> Unit) = Button(this).apply {
        id = idValue
        text = label
        contentDescription = label
        isAllCaps = false
        setOnClickListener { action() }
    }

    private fun savedSort(key: String): SortMode = runCatching {
        SortMode.valueOf(preferences.getString(key, SortMode.NAME.name) ?: SortMode.NAME.name)
    }.getOrDefault(SortMode.NAME)

    private fun sortedProducts(mode: SortMode): List<Product> = when (mode) {
        SortMode.NAME -> products.sortedBy { it.name }
        SortMode.PRICE -> products.sortedBy { it.cents }
    }

    private fun price(cents: Int) = "$${"%.2f".format(cents / 100.0)}"

    private fun weightedParams() = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val ASYNC_DELAY_MS = 700L
        const val MATCH_PARENT = ViewGroup.LayoutParams.MATCH_PARENT
        const val WRAP_CONTENT = ViewGroup.LayoutParams.WRAP_CONTENT
    }
}
