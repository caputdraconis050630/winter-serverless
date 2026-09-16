// Java service: JDK built-in HTTP server, JVM startup is the realistic cold cost.
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

public class Main {
    public static void main(String[] args) throws Exception {
        long t0 = System.currentTimeMillis();
        // Simulate framework init: class loading + some CPU work
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] acc = new byte[]{0};
        for (int i = 0; i < 20000; i++) acc = md.digest(acc);
        double initSec = (System.currentTimeMillis() - t0) / 1000.0;

        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", exchange -> {
            String body = String.format(
                "{\"runtime\":\"java-svc\",\"init_sec\":%.3f}", initSec);
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
    }
}
