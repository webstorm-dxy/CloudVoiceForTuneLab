using TuneLab.CloudVoice.Voice;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class LyricPhonemesTests
{
    [Fact]
    public void LatinSyllable_SplitsLeadingVowelAndCoda()
    {
        var syllable = LyricPhonemes.Predict("la");

        Assert.False(syllable.Rest);
        Assert.Equal(["l"], syllable.Leading.Select(p => p.Symbol));
        Assert.Equal(["a"], syllable.Body.Select(p => p.Symbol));
        Assert.Equal(0.07, syllable.BodyOffset, 6);
        Assert.Equal(0, syllable.Leading[0].StretchWeight);
        Assert.Equal(1, syllable.Body[0].StretchWeight);
    }

    [Fact]
    public void ToneDigit_IsStripped()
    {
        var syllable = LyricPhonemes.Predict("ma3");

        Assert.Equal(["m"], syllable.Leading.Select(p => p.Symbol));
        Assert.Equal(["a"], syllable.Body.Select(p => p.Symbol));
    }

    [Fact]
    public void ConsonantCluster_And_MultipleVowels()
    {
        var syllable = LyricPhonemes.Predict("krait");

        Assert.Equal(["k", "r"], syllable.Leading.Select(p => p.Symbol));
        Assert.Equal(["a", "i", "t"], syllable.Body.Select(p => p.Symbol));
        Assert.Equal(0.14, syllable.BodyOffset, 6);
    }

    [Fact]
    public void ConsonantOnly_LandsInBody()
    {
        var syllable = LyricPhonemes.Predict("st");

        Assert.Empty(syllable.Leading);
        Assert.Equal(["s", "t"], syllable.Body.Select(p => p.Symbol));
        Assert.All(syllable.Body, phoneme => Assert.Equal(0, phoneme.StretchWeight));
    }

    [Fact]
    public void NonLatin_KeepsWholeTextAsBody()
    {
        var syllable = LyricPhonemes.Predict("漢");

        Assert.False(syllable.Rest);
        Assert.Equal(["漢"], syllable.Body.Select(p => p.Symbol));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("-")]
    [InlineData("R")]
    public void Rests_AreRecognized(string lyric)
    {
        Assert.True(LyricPhonemes.Predict(lyric).Rest);
    }
}
