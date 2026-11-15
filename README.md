# Lord Billy – »Oye Como Wau«

Ein 45-Sekunden-Musikvideo im Stil des Musikfernsehens. Darin singt
**Billy**, unser Chow-Chow (Rüde, 6 Jahre), mit tiefer Männerstimme ins
Mikrofon. Der Song ist Latin-Rock à la Santana, mit Am7–D9-Vamp,
Cha-Cha-Groove, Congas, Hammond-Orgel und singender Leadgitarre.

<p align="center"><img src="video/poster.jpg" width="320" alt="Billy singt ins Mikrofon"></p>

▶️ **Video:** [`video/LordBilly_OyeComoWau.mp4`](video/LordBilly_OyeComoWau.mp4)
(1080×1920 Hochformat, 30 fps, 45 s, mit Ton)

## Inhalt

| Pfad | Was |
|---|---|
| `video/LordBilly_OyeComoWau.mp4` | das fertige Musikvideo mit eingebrannten Karaoke-Untertiteln |
| `video/LordBilly_OyeComoWau.srt` | die gesungenen Zeilen zusätzlich als Untertiteldatei |
| `video/poster.jpg` | Vorschaubild |
| `audio/LordBilly_OyeComoWau.mp3` | nur der Song |
| `bilder/` | die drei Originalfotos von Billy (aus Google Drive › GitHub › LordBilly) |
| `src/` | Programme, mit denen Song und Video erzeugt werden |

## Das Video

- **Gesang:** Billy singt mit einer tiefen Männerstimme im Bariton (A2–A3).
  Sein Unterkiefer bewegt sich synchron zu jedem gesungenen Laut: Bei
  »a« und »o« geht das Maul weit auf, bei »m« und »b« schließt es sich.
- **Mikrofon:** Vor dem Maul steht ein Gesangsmikrofon auf einem Ständer. Es
  folgt jeder Kamerabewegung und wirft einen Schatten auf das Fell.
- **Musikfernsehen-Look:** Sender-Logo »WAU TV« oben rechts, »Weltpremiere«
  im Intro, Credits unten links am Anfang und am Ende, harte Schnitte im
  Takt, Bühnen-Lichtkegel, RGB-Versatz und Kamerawackler auf dem Beat,
  Stroboskop und Farbstimmungen in Refrain und Solo, Split-Screen mit allen
  drei Billys gleichzeitig, große Wörter im Refrain, Filmkorn
- **Untertitel:** Unten im Bild steht immer die Zeile, die Billy gerade
  singt. Die schon gesungenen Silben färben sich dabei gold (Karaoke-Stil).

Ablauf: Intro (0–4 s) · Gitarren-Intro und Einsingen (4–12 s) · Strophe
(12–28 s) · Refrain (28–36 s) · Gitarrensolo (36–42 s) · Finale (42–45 s)

## Liedtext

> Oye, mein Name ist Billy,
> ein Chow Chow mit Löwenmähne.
> Sechs Jahre Sonne im Fell,
> die Zunge so blau wie die Nacht.
>
> Oye como wau, wau-wau!
> Lord Billy singt für dich – auuu!

## Hinweis zur Musik

Ein echter Santana-Song ist urheberrechtlich geschützt. Text und Melodie
dürfen hier nicht nachgebaut werden. Melodie, Text und Arrangement sind
deshalb eine **Eigenkomposition im Santana-Stil**. Das Wortspiel im Titel
ist eine Hommage. Alle Klänge entstehen per Synthese in `src/make_song.py`.
Die Gesangsstimme stammt aus der freien Sprachsynthese MBROLA (deutsche
Männerstimmen de6 und de4), Ton für Ton auf die Melodie gestimmt. Es wurden
keine fremden Aufnahmen verwendet.

## Selbst neu erzeugen

Voraussetzungen: Python 3 mit `numpy`, `scipy`, `opencv-python-headless`,
`Pillow`, außerdem `ffmpeg`, die Schrift *Inter* und MBROLA mit den
deutschen Männerstimmen.

```bash
sudo apt-get install mbrola mbrola-de6 mbrola-de4 ffmpeg fonts-inter
pip install numpy scipy opencv-python-headless Pillow
mkdir -p build
python3 src/make_song.py build/song.wav
python3 src/make_video.py build/song.wav video/LordBilly_OyeComoWau.mp4
```

- `src/timeline.py`: Tempo, Melodie, Liedtext als Phonemfolge, Gitarrenlinie
  und die daraus berechnete Maulöffnung. Ton und Bild nutzen denselben
  Zeitplan.
- `src/make_song.py`: Percussion, Bass, Orgel, Gitarre, Gesang (MBROLA,
  gedoppelt, mit Sub-Oktave) und der Mix
- `src/make_video.py`: Kamerafahrten über die Fotos, Unterkiefer-Animation,
  Mikrofon, Musikfernsehen-Effekte, Titel und Karaoke-Untertitel

Einzelne Vorschaubilder: `python3 src/make_video.py --preview 13.1,29.3 vorschau/`
