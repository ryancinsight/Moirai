### Removed
- The non-functional `UniversalChannel`/`UniversalSender`/`UniversalReceiver`
  placeholders (their `send`/`recv` ignored their argument and returned `Closed`;
  a channel generic over an arbitrary `Send` `T` cannot serialize for transport).
  The working archive channels (above) are their realized replacement.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
