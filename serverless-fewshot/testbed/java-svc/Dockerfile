FROM eclipse-temurin:17-jdk-jammy
WORKDIR /app
COPY Main.java .
RUN javac Main.java
ENV PORT=8080
EXPOSE 8080
CMD ["java", "-Xshare:off", "Main"]
